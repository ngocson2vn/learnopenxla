# Toy Model
For learning AutoFusion

## Setup
```Bash
# python 3.7.3
sudo apt install ca-certificates
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec bash -l

sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv install 3.7.3
pyenv global 3.7.3

# /opt/tiger/auto_fusion_sdk
bvc clone erdos/operators/tools --version 1.0.0.337 /opt/tiger/auto_fusion_sdk

# Install tensorsharp
bvc clone aml/erdos/tensorsharp --version 1.0.0.2408 /tmp/tensorsharp
pip3 install /tmp/tensorsharp/bytedance.tensorsharp-0.1.0.20-py2.py3-none-any.whl
```

## Create Model
```Bash
./create.sh
```

## Apply AutoFusion
```Bash
./apply.sh
```

## Run Model
```Bash
./run.sh
```

## Debug
https://code.byted.org/son.nguyen/auto_fusion_laniakea