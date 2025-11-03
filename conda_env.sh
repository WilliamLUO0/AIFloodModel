srun --pty -A uoa04425 -p milan,genoa --gres=gpu:1 -t 01:00:00 -c 4 --mem=16G bash -l
srun --pty -A uoa04425 -p milan,genoa -t 01:00:00 -c 4 --mem=8G bash -l

module purge
module load Miniconda3/23.10.0-1
module load CUDA/12.1.1
module load GCC/12.3.0

eval "$(conda shell.bash hook)"
mkdir -p "$(dirname "/nesi/project/uoa04425/zluo784/envs/pft39")"
conda create -y -p "/nesi/project/uoa04425/zluo784/envs/pft39" python=3.9
conda activate "/nesi/project/uoa04425/zluo784/envs/pft39"

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

cd /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel
pip install -r requirements.txt
python setup.py develop

cd ops_smm
python setup.py install

python - <<'PY'
import torch, smm_cuda, basicsr
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
print("basicsr ok:", basicsr is not None)
print("basicsr loaded from:", basicsr.__file__)
print("smm_cuda ok:", smm_cuda is not None)
print("smm_cuda loaded from:", smm_cuda.__file__)
PY

pip show basicsr
pip show smm_cuda
