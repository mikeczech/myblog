+++
title = 'Implementing a Python Package Repository on GCP'
date = 2020-09-13T12:06:21+02:00
draft = false
tags = ["cloud", "gcp", "python"]
+++

**Update (March 14, 2021):** *Google has recently announced that their Artifact Registry now also supports PyPI repositories in its alpha stage (refer to [Artifact Registry documentation](https://cloud.google.com/artifact-registry/docs/python/quickstart) for more information). While I haven't had the opportunity to test this feature personally, it appears to be a promising development. Therefore, I recommend considering the use of Artifact Registry for this purpose.*

In my daily work, I often encounter challenges that seem so fundamental, I assume ready-made solutions must exist. One such example is setting up a private repository for Python packages on the Google Cloud Platform (GCP). At first glance, this task might appear straightforward, especially considering that many companies produce proprietary source code which they need to distribute internally. However, the devil is in the details.

The ideal private repository should meet several key criteria:

1. Authentication should be seamlessly integrated with Google's [IAM](https://cloud.google.com/iam), eliminating the need for separate credentials for public endpoints.
2. Compatibility with managed services like [Google Cloud Build](https://cloud.google.com/cloud-build) and popular dependency management tools, including [Poetry](https://github.com/python-poetry/poetry), [pip](https://pip.pypa.io/en/stable/), and [Conda](https://docs.conda.io/en/latest/), is crucial.
3. The repository should utilize a [Google Cloud Storage](https://cloud.google.com/storage) bucket for benefits like unlimited storage, backup capabilities, and ease of access.

Most online solutions I encountered required an additional layer of authentication, introducing potential vulnerabilities into the infrastructure. To address this, my colleague and I experimented with a different approach. This setup uses a pod in the [Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine), port forwarding, [NGINX](https://www.nginx.com/), and [gcsfuse](https://github.com/GoogleCloudPlatform/gcsfuse). Note that this architecture can be adapted to the [Compute Engine](https://cloud.google.com/compute) using [IAP TCP Forwarding](https://cloud.google.com/iap/docs/tcp-forwarding-overview) if GKE is not your preferred choice.

In this design, users initiate a port forwarding process that directs package requests to the pod. The pod, in turn, delivers the contents of a GCS bucket, structured according to [pip's requirements](https://packaging.python.org/guides/hosting-your-own-index/), using NGINX and gcsfuse. 

Let's delve into the specifics. This guide presupposes the existence of a GKE cluster and a GCS bucket named **your-project-pypapo**, which contains the repository's contents.
The initial step involves creating a Docker image capable of mounting the bucket with gcsfuse and serving its contents via NGINX:

```docker
FROM google/cloud-sdk:305.0.0-slim
ARG PROJECT
ENV PROJECT ${PROJECT}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget autofs nginx \
  && echo "deb http://packages.cloud.google.com/apt cloud-sdk-xenial main" | tee /etc/apt/sources.list.d/google-cloud.sdk.list \
  && echo "deb http://packages.cloud.google.com/apt gcsfuse-xenial main" | tee /etc/apt/sources.list.d/gcsfuse.list \
  && wget -qO- https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update && apt-get install -y --no-install-recommends google-cloud-sdk gcsfuse && apt-get clean \
  && mkdir -p /etc/autofs && touch /etc/autofs/auto.gcsfuse && rm -rf /var/lib/apt/lists \
  && useradd -ms /bin/bash  pypapo \
  && mkdir /repo-data \
  && chown -R pypapo /repo-data \
  && mkdir -p /var/log/nginx \
  && chown -R pypapo /var/log/nginx

USER pypapo
ENTRYPOINT ["bash", "-c", "nginx && gcsfuse --implicit-dirs --foreground ${PROJECT}-pypapo /repo-data"]
```

To build the image, we use [Cloud Build](https://cloud.google.com/cloud-build) with the following configuration (**cloudbuild.yaml**):

```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: [
    'build',
    '-t', 'eu.gcr.io/$PROJECT_ID/pypapo:latest',
    '--build-arg', 'PROJECT=$PROJECT_ID',
    '-f', 'Dockerfile',
    '.'
    ]
images: ['eu.gcr.io/$PROJECT_ID/pypapo:latest']
```

Then submit it:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

After building the image, we proceed to define a Kubernetes Deployment and Service using the built image (**pypapo.yaml**):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: pypapo-server
spec:
  type: NodePort
  ports:
  - name: web
    port: 3333
    targetPort: 3333
    protocol: TCP
  selector:
    app: pypapo
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pypapo
  namespace: pypapo
spec:
  selector:
    matchLabels:
      app: pypapo
  template:
    metadata:
      labels:
        app: pypapo
    spec:
      serviceAccount: pypapo
      containers:
      - name: pypapo
        image: "eu.gcr.io/your-project/pypapo:latest"
        imagePullPolicy: Always
        resources:
          limits:
            memory: "128Mi"
            cpu: "100m"
        securityContext:
          privileged: true # due to gcsfuse
        ports:
        - containerPort: 3333
        volumeMounts:
          - mountPath: /etc/nginx
            readOnly: true
            name: nginx-conf
      volumes:
        - name: nginx-conf
          configMap:
            name: nginx-conf
            items:
              - key: nginx.conf
                path: nginx.conf
---
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: pypapo
  name: nginx-conf
data:
  nginx.conf: |
    error_log /tmp/error.log;
    pid       /tmp/nginx.pid;

    events {}
    http {
      server {
        listen       3333;

        access_log /tmp/nginx_host.access.log;
        client_body_temp_path /tmp/client_body;
        fastcgi_temp_path /tmp/fastcgi_temp;
        proxy_temp_path /tmp/proxy_temp;
        scgi_temp_path /tmp/scgi_temp;
        uwsgi_temp_path /tmp/uwsgi_temp;

        # Serve local files
        location / {
          root /repo-data;
          autoindex on;
        }
      }
    }
```

These configurations are then applied using **kubectl**.

```bash
kubectl apply -f pypapo.yaml
```

Once the service is up and running, you can use tools like [Poetry](https://github.com/python-poetry/poetry) to install packages from this repository. This requires modifying the **pyproject.toml** file to include an additional package source pointing to the local repository service.

```toml
[tool.poetry]
name = "myproj"
version = "0.1.0"
description = ""
authors = ["mike"]

[tool.poetry.dependencies]
python = "^3.7"
my-package = "^0.14.1"

[[tool.poetry.source]]
name = "foo"
url = "http://localhost:3333"

[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
```

Finally, establish port forwarding and proceed with package installation:

```bash
kubectl port-forward service/pypapo-server 3333:3333 # run this in a separate shell
poetry install
```

This approach facilitates the efficient distribution of Python packages within projects. However, it's important to be aware of certain limitations. Using gcsfuse requires the repository service to run in a privileged security context, which can pose a security risk. Additionally, users need appropriate permissions to establish port forwarding, another consideration from a security standpoint. Despite these challenges, this solution has proven effective for our needs, and we aim to refine its architecture in the near future.
