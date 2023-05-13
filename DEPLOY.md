## Deployment Instructions

### Install Docker

Install Docker and docker-compose for whatever system you are going to be using for hosting. Instructions for a few common Linux distributions are provided below:

#### Fedora

```bash
sudo dnf install moby-engine docker-compose
```

#### Ubuntu

```bash
sudo apt update
sudo apt install docker.io docker-compose
```

### Run Application

Once docker is installed, simply run the following commands to build the application container and run the application:

```bash
docker-compose build
docker-compose up -d
```

Once the application is running, it will be accessible at <http://127.0.0.1:8050> on the system.

