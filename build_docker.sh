cp dockerfiles/.dockerignore .dockerignore
docker build -t cuda117_tensort:22_12 -f dockerfiles/Dockerfile.tensorrt22_12 .