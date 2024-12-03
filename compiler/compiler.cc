/* Copyright 2022 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference:
https://github.com/openxla/xla/tree/main/xla/examples/axpy
==============================================================================*/

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <mutex>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/platform_util.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

// #include "xla/stream_executor/cuda/cuda_platform.h"
// #include "xla/stream_executor/platform.h"

std::mutex global_log_mutex;
#define GLOBAL_LOCK() std::lock_guard<std::mutex> lk(global_log_mutex)

#define LOG_ERROR(...) \
do { \
  GLOBAL_LOCK(); \
  fprintf(stderr, "ERROR %s:%d: ", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\n"); \
} while(0)

#define LOG_INFO(...) \
do { \
  GLOBAL_LOCK(); \
  fprintf(stderr, "INFO "); \
  fprintf(stdout, __VA_ARGS__); \
  fprintf(stdout, "\n"); \
} while(0)

namespace xla {

absl::Status compile() {
  // Setup client
  LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();

  // Retrieve the "platform" we intend to execute the computation on. The
  // concept of "platform" in XLA abstracts entirely everything needed to
  // interact with some hardware (compiler, runtime, etc.). New HW vendor
  // plugs into XLA by registering a new platform with a different string
  // key. For example for an Nvidia GPU change the following to:
  //   PlatformUtil::GetPlatform("CUDA"));
  TF_ASSIGN_OR_RETURN(
    se::Platform* platform, PlatformUtil::GetPlatform("Host"));
  TF_ASSIGN_OR_RETURN(
    se::StreamExecutor* executor, platform->ExecutorForDevice(/*ordinal=*/0));

  // LocalDeviceState and PjRtStreamExecutorDevice describes the state of a
  // device which can do computation or transfer buffers. This could represent a
  // GPU or accelerator.
  auto device_state = std::make_unique<LocalDeviceState>(
      executor, local_client, LocalDeviceState::kSynchronous,
      /*max_inflight_computations=*/32,
      /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
  auto device = std::make_unique<PjRtStreamExecutorDevice>(
      0, std::move(device_state), "cpu");
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices.emplace_back(std::move(device));

  // The PjRtStreamExecutorClient will allow us to compile and execute
  // computations on the device we just configured.
  auto pjrt_se_client =
      PjRtStreamExecutorClient("cpu", local_client, std::move(devices),
                               /*process_index=*/0, /*allocator=*/nullptr,
                               /*host_memory_allocator=*/nullptr,
                               /*should_stage_host_to_device_transfers=*/false,
                               /*cpu_run_options=*/nullptr);

  // Read StableHLO program to string.
  std::string program_path = "sample/frozen_graph.stablehlo";
  std::string program_string;
  TF_RETURN_IF_ERROR(
    tsl::ReadFileToString(tsl::Env::Default(), program_path, &program_string)
  );

  std::cerr << "Loaded StableHLO program from " << program_path << ":\n"
            << program_string << std::endl;

  // Register MLIR dialects necessary to parse our program. In our case this is
  // just the Func dialect and StableHLO.
  mlir::DialectRegistry dialects;
  mlir::stablehlo::registerAllDialects(dialects);
  dialects.insert<mlir::func::FuncDialect>();
  dialects.insert<mlir::shape::ShapeDialect>();

  // Parse StableHLO program.
  auto ctx = std::make_unique<mlir::MLIRContext>(dialects);
  mlir::OwningOpRef<mlir::ModuleOp> program =
      mlir::parseSourceString<mlir::ModuleOp>(program_string, ctx.get());

  // Use our client to compile our StableHLO program to an executable.
  auto status_or = pjrt_se_client.Compile(*program, CompileOptions{});
  if (!status_or.ok()) {
    LOG_ERROR(status_or.status().ToString().c_str());
    return status_or.status();
  }
  std::unique_ptr<PjRtLoadedExecutable> executable = std::move(status_or.value());

  // Create inputs to our computation.
  auto x1_literal = xla::LiteralUtil::CreateR2F32Linspace(0.0f, 1.0f, 3, 4);
  auto x2_literal = xla::LiteralUtil::CreateR2F32Linspace(0.0f, 1.0f, 3, 4);

  std::cerr << "Computation inputs:" << std::endl;
  std::cerr << "\tx1:" << x1_literal << std::endl;
  std::cerr << "\tx2:" << x2_literal << std::endl;

  // Get the host device.
  PjRtDevice* cpu = pjrt_se_client.devices()[0];

  // Transfer our literals to buffers. If we were using a GPU, these buffers
  // would correspond to device memory.
  TF_ASSIGN_OR_RETURN(
    std::unique_ptr<PjRtBuffer> x1,
    pjrt_se_client.BufferFromHostLiteral(x1_literal, cpu));
  TF_ASSIGN_OR_RETURN(
    std::unique_ptr<PjRtBuffer> x2,
    pjrt_se_client.BufferFromHostLiteral(x2_literal, cpu));

  // Do our computation.
  TF_ASSIGN_OR_RETURN(
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
    executable->Execute({{x1.get(), x2.get()}}, /*options=*/{}));

  // Convert result buffer back to literal.
  TF_ASSIGN_OR_RETURN(
    std::shared_ptr<Literal> result_literal,
    result[0][0]->ToLiteralSync());

  std::cerr << "Computation output: " << *result_literal << std::endl;
  return absl::OkStatus();
}

}  // namespace xla

int main(int argc, char** argv) {
  auto status = xla::compile();
  if (!status.ok()) {
    std::cerr << status.ToString() << "\n";
    std::cerr << "FAILED!\n";
  }
}