cd build
make main
cd ..
./build/bin/main -m ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -p "once upon a time" -n 64 --vram-budget 10
