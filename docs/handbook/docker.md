---
title: Docker Commands Handbook # 标题
date: 2025-04-05 18:00:00
---

## Docker 命令

```bash
(base) S:\SystemDefault\Desktop\kgrag>docker --help

Usage:  docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  build       Build an image from a Dockerfile
  pull        Download an image from a registry
  push        Upload an image to a registry
  images      List images
  login       Log in to a registry
  logout      Log out from a registry
  search      Search Docker Hub for images
  version     Show the Docker version information
  info        Display system-wide information

Management Commands:
  builder     Manage builds
  buildx*     Docker Buildx (Docker Inc., v0.12.1-desktop.4)
  compose*    Docker Compose (Docker Inc., v2.24.5-desktop.1)
  container   Manage containers
  context     Manage contexts
  debug*      Get a shell into any image or container. (Docker Inc., 0.0.24)
  dev*        Docker Dev Environments (Docker Inc., v0.1.0)
  extension*  Manages Docker extensions (Docker Inc., v0.2.21)
  feedback*   Provide feedback, right in your terminal! (Docker Inc., v1.0.4)
  image       Manage images
  init*       Creates Docker-related starter files for your project (Docker Inc., v1.0.0)   
  manifest    Manage Docker image manifests and manifest lists
  network     Manage networks
  plugin      Manage plugins
  sbom*       View the packaged-based Software Bill Of Materials (SBOM) for an image (Anchore Inc., 0.6.0)
  scout*      Docker Scout (Docker Inc., v1.4.1)
  system      Manage Docker
  trust       Manage trust on Docker images
  volume      Manage volumes

Swarm Commands:
  swarm       Manage Swarm

Commands:
  attach      Attach local standard input, output, and error streams to a running container 
  commit      Create a new image from a container's changes
  cp          Copy files/folders between a container and the local filesystem
  create      Create a new container
  diff        Inspect changes to files or directories on a container's filesystem
  events      Get real time events from the server
  export      Export a container's filesystem as a tar archive
  history     Show the history of an image
  import      Import the contents from a tarball to create a filesystem image
  inspect     Return low-level information on Docker objects
  kill        Kill one or more running containers
  load        Load an image from a tar archive or STDIN
  logs        Fetch the logs of a container
  pause       Pause all processes within one or more containers
  port        List port mappings or a specific mapping for the container
  rename      Rename a container
  restart     Restart one or more containers
  rm          Remove one or more containers
  rmi         Remove one or more images
  save        Save one or more images to a tar archive (streamed to STDOUT by default)      
  start       Start one or more stopped containers
  stats       Display a live stream of container(s) resource usage statistics
  stop        Stop one or more running containers
  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE
  top         Display the running processes of a container
  unpause     Unpause all processes within one or more containers
  update      Update configuration of one or more containers
  wait        Block until one or more containers stop, then print their exit codes

Global Options:
      --config string      Location of client config files (default
                           "C:\\Users\\DELL\\.docker")
  -c, --context string     Name of the context to use to connect to the
                           daemon (overrides DOCKER_HOST env var and
                           default context set with "docker context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info",
                           "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")

Global Options:
      --config string      Location of client config files (default
                           "C:\\Users\\DELL\\.docker")
  -c, --context string     Name of the context to use to connect to the
                           daemon (overrides DOCKER_HOST env var and
                           default context set with "docker context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info",
                           "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
      --config string      Location of client config files (default
                           "C:\\Users\\DELL\\.docker")
  -c, --context string     Name of the context to use to connect to the
                           daemon (overrides DOCKER_HOST env var and
                           default context set with "docker context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info",
                           "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
                           daemon (overrides DOCKER_HOST env var and
                           default context set with "docker context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info",
                           "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info",
                           "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
                           "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
      --tlscacert string   Trust certs signed only by this CA (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
      --tlscert string     Path to TLS certificate file (default
                           "C:\\Users\\DELL\\.docker\\ca.pem")
      --tlscert string     Path to TLS certificate file (default
      --tlscert string     Path to TLS certificate file (default
                           "C:\\Users\\DELL\\.docker\\cert.pem")
      --tlskey string      Path to TLS key file (default
                           "C:\\Users\\DELL\\.docker\\key.pem")
      --tlsverify          Use TLS and verify the remote
  -v, --version            Print version information and quit

Run 'docker COMMAND --help' for more information on a command.

For more help on how to use Docker, head to https://docs.docker.com/go/guides/
```

## 一个处理案例：拉取并进入容器

```bash
# Pull the docker-compose.yml file
$ curl -sSL https://raw.githubusercontent.com/XXX/master/dev/release/docker-compose.yml -o docker-compose.yml
# Start the services
$ docker compose -f docker-compose.yml up
# check the port
$ netstat -ano | findstr "3306"
$ docker ps

CONTAINER ID   IMAGE                                                                COMMAND                   CREATED       STATUS          PORTS                                                           NAMES
fffc6e176abe   server:latest   "java -jar arks-sofa…"   12 days ago   Up 54 minutes   0.0.0.0:8887->8887/tcp                                          release-server

$ docker exec -it fffc6e176abe /bin/bash # 进入容器

$ find / -name "*.html" 2>/dev/null # 搜索所有以 .html 结尾的文件

/usr/share/doc/python3.8/python-policy.html # python-policy.html：Python 的政策和规范文档。
/usr/share/doc/shared-mime-info/shared-mime-info-spec.html # shared-mime-info-spec.html：MIME 类型规范文档。
/usr/share/doc/python3/python-policy.html # python-policy.html：Python 3 的通用政策文档。
/XXXXXX_venv/lib/python3.8/site-packages/torch/utils/model_dump/skeleton.html # skeleton.html：PyTorch 模型结构的可视化文件。

$ find / -name "*.jar" 2>/dev/null # 搜索所有以 .jar 结尾的文件

/usr/lib/jvm/java-11-openjdk-amd64/lib/jrt-fs.jar
/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/management-agent.jar
...
/usr/share/java/maven3-compat.jar
/arks-sofaboot-0.0.1-SNAPSHOT-executable.jar

$ jar xf arks-sofaboot-0.0.1-SNAPSHOT-executable.jar

$ docker cp fffc6e176abe:/BOOT-INF "S:\SystemDefault\Desktop\XXXXXX"

$ docker cp "S:\SystemDefault\Desktop\XXXXXX\BOOT-INF" fffc6e176abe:/
```

BOOT-INF/classes/static/umi.1e986a90.js

```bash
$ jar uf arks-sofaboot-0.0.1-SNAPSHOT-executable.jar umi.1e986a90.js
$ jar uf arks-sofaboot-0.0.1-SNAPSHOT-executable.jar BOOT-INF/classes/static/umi.1e986a90.js
$ cat BOOT-INF/classes/umi.1e986a90.js
```
