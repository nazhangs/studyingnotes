# Docker

dockerd -H tcp://127.0.0.1:2375 -H unix:///var/run/docker.sock &

docker exec it <containerDorcName> /bin/bash
>if the container was not started by /bin/bash, then use exec to create a bash instance inside the container

docker attach <containerDorName>
>when docker container was started using /bin/bash command, then the container can be attaced. use command line in container

docker run -p 4000:80 image

docker build -f docker-file -t name:tag <context>
> build docker image

docker images --fileter "dangline=true"

docker rmi $(docker images -f "dangline=true" -q)

docker images --format "{{.ID}}:{{.Repository}}"
docker images --format "table{{.ID}}\t{{.Repository}}\t{{.Tag}}"

container:
http://0.0.0.0
