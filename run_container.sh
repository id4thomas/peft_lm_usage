CONTAINER="tensorrt_22_12"

# docker run -it -d \
#     --name ${CONTAINER} \
#     --gpus device=0 \
#     -v ${pwd}/experiments:/usr/app/experiments \
#     -v ${pwd}/src:/usr/app/src \
#     -v ${pwd}/sample_data:/usr/app/sample_data \
#     cuda117_tensort:22_12

docker run -it -d \
    --name ${CONTAINER} \
    --gpus device=0 \
    --mount type=bind,source=${PWD}/experiments,target=/usr/app/experiments \
    --mount type=bind,source=${PWD}/src,target=/usr/app/src \
    --mount type=bind,source=${PWD}/sample_data,target=/usr/app/sample_data \
    cuda117_tensort:22_12