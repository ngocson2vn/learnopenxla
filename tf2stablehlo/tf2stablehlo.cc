/* Reference: 
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/tensorflow_to_stablehlo
==============================================================================*/

#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_import.h"

using namespace tensorflow;
namespace tfq = tensorflow::quantization;

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


namespace {

using llvm::cl::opt;

// NOLINTNEXTLINE
opt<std::string> input_filename(llvm::cl::Positional,
                            llvm::cl::desc("<input path>"), llvm::cl::Required);

// NOLINTNEXTLINE
opt<std::string> output_filename("o", llvm::cl::desc("<output path>"),
                                 llvm::cl::Optional, llvm::cl::init("-"));

}  // namespace

namespace mlir {

namespace {
// Convert an TF module to a StableHLO module
absl::StatusOr<OwningOpRef<ModuleOp>> ConvertMlirToStablehlo(
    mlir::OwningOpRef<mlir::ModuleOp> module_op, MLIRContext* context) {
  std::string mlir_dump_file_prefix = "";
  mlir::PassManager pass_manager(context);
  tfq::AddTFToStablehloPasses(pass_manager);

  auto status = tfq::RunPassesOnModuleOp(
      absl::StrCat(mlir_dump_file_prefix, "_post_stablehlo_passes"),
      pass_manager, *module_op);

  if (!status.ok()) {
    LOG_ERROR(status.ToString().c_str());
    return status;
  }

  return std::move(module_op);
}

// Dump the ModuleOp 'module' to the file specified using 'outputFileName'
absl::Status ExportModule(ModuleOp module, std::string ofname) {
  std::string error_msg;
  auto output = openOutputFile(ofname, &error_msg);
  if (output == nullptr) {
    return absl::AbortedError(
        absl::StrCat("Unable to write to output path: ", error_msg));
  }

  // Export StableHLO MLIR as output
  std::string result;
  llvm::raw_string_ostream os(result);
  OpPrintingFlags printing_flags;
  module.print(os, printing_flags);
  os.flush();

  output->os() << result;
  output->keep();

  return absl::OkStatus();
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TF Saved Model to Stablehlo converter\n");

  mlir::DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string error_message;
  std::unique_ptr<llvm::MemoryBuffer> input = mlir::openInputFile(input_filename, &error_message);
  if (!input) {
    LOG_ERROR(error_message.c_str());
    return 1;
  }

  tensorflow::GraphDef input_graph_def;
  tensorflow::Status status = tensorflow::LoadProtoFromBuffer(
      {input->getBuffer().data(), input->getBuffer().size()}, &input_graph_def);
  if (!status.ok()) {
    LOG_ERROR("loading input graph to graph def failed.");
    return 1;
  }

  std::vector<std::string> input_array_vector;
  status = tensorflow::ParseNodeNames(input_arrays, input_array_vector);
  if (!status.ok()) {
    LOG_ERROR("parse input names error.");
    return 1;
  }

  std::vector<std::string> input_dtype_vector;
  status = tensorflow::ParseNodeDataTypes(input_dtypes, input_dtype_vector);
  if (!status.ok()) {
    LOG_ERROR("parse input dtypes error.");
    return 1;
  }

  std::vector<std::optional<std::vector<int>>> input_shapes_vector;
  status = tensorflow::ParseNodeShapes(input_shapes, input_shapes_vector);
  if (!status.ok()) {
    LOG_ERROR("parse input shapes error.");
    return 1;
  }

  std::vector<std::string> output_array_vector;
  status = tensorflow::ParseNodeNames(output_arrays, output_array_vector);
  if (!status.ok()) {
    LOG_ERROR("parse output node error.");
    return 1;
  }

  tensorflow::GraphdefToMlirOptions import_options;
  import_options.prune_unused_nodes = true;
  import_options.convert_legacy_fed_inputs = false;
  import_options.graph_as_function = graph_as_function;
  import_options.upgrade_legacy = false;
  import_options.enable_shape_inference = true;
  import_options.unconditionally_use_set_output_shapes = true;
  auto module_or = tensorflow::GraphdefToMlirTranslateFunction(
    input_graph_def.SerializeAsString(),
    input_array_vector, input_dtype_vector, input_shapes_vector,
    output_array_vector, {},
    import_options,
    &context
  );

  if (!module_or.status().ok()) {
    LOG_ERROR(module_or.status().ToString().c_str());
    return module_or.status().raw_code();
  }

  mlir::OwningOpRef<mlir::ModuleOp> &module = module_or.value();
  std::vector<std::string> names_vector = absl::StrSplit(
                           input_filename, '.', absl::SkipEmpty());
  std::string mlir_output_path = names_vector[0] + ".mlir";
  auto export_status = mlir::ExportModule(*module, mlir_output_path);
  if (!export_status.ok()) {
    LOG_ERROR(export_status.ToString().c_str());
    return export_status.raw_code();
  }

  module_or = mlir::ConvertMlirToStablehlo(std::move(module), &context);
  if (!module_or.ok()) {
    LOG_ERROR(module_or.status().ToString().c_str());
    return module_or.status().raw_code();
  }

  export_status = mlir::ExportModule(module_or->get(), output_filename);
  if (!export_status.ok()) {
    LOG_ERROR(export_status.ToString().c_str());
    return export_status.raw_code();
  }

  return 0;
}
