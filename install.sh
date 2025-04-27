git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
mkdir repositories
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion-stability-ai
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
git clone https://github.com/salesforce/BLIP.git repositories/BLIP
cd repositories/stable-diffusion-stability-ai
git checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf
cd ../BLIP
git checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9