+++
title = 'Optimizing Resource Utilization in Batch Jobs on GKE'
date = 2019-04-15T21:43:57+01:00
draft = false
+++

In our team, we frequently execute batch jobs that demand specialized hardware, particularly model training tasks that typically require multiple GPUs for timely completion. To facilitate these jobs, we employ the [*Google Kubernetes Engine* (GKE)](https://cloud.google.com/kubernetes-engine/), a managed service on the Google Cloud Platform (GCP). Our workflow involves submitting a job to Kubernetes, which then allocates necessary resources, executes the job, and subsequently deallocates these resources to minimize costs. Typically, Kubernetes procures additional resources by provisioning new compute nodes and integrating them into the cluster.

GKE streamlines this process, though the intricacies can sometimes be challenging. Previously, we encountered issues where Kubernetes would not remove seemingly idle nodes from the cluster, leading to increased and regular cloud expenses. To understand why our nodes were not being released, let’s examine a typical GKE configuration using [Terraform](https://www.terraform.io/):

```hcl
provider "google-beta" {
  project = "myproject"
  region  = "europe-west1"
}

resource "google_container_cluster" "primary" {
  name               = "my-cluster"
  initial_node_count = 2
  subnetwork         = "default-europe-west1"

  addons_config {
    kubernetes_dashboard {
      disabled = true
    }
  }

  node_config {
    machine_type = "n1-standard-1"
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    disk_size_gb = 20
  }
}

resource "google_container_node_pool" "model-training-np" {
  provider           = "google-beta"
  name               = "training-np"
  cluster            = "my-cluster"
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }

  node_config {
    machine_type = "n1-standard-32"
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    disk_size_gb = 100

    guest_accelerator {
      type  = "nvidia-tesla-k80"
      count = 4
    }
  }
}
```

This configuration establishes a GKE cluster with a default node pool comprising two compute nodes, each equipped with one vCPU and approximately 4GB of RAM, sufficient for running GKE system services. Additionally, it includes a separate node pool for model training with an initial count of zero. These on-demand nodes are each outfitted with 32 vCPUs, 120GB of RAM, and 4 GPUs. Autoscaling is configured to allow a maximum of three nodes, enabling Kubernetes to provision up to three nodes as needed for job execution. To deploy this configuration in your GCP project, follow these steps:

```bash
gcloud auth application-default login # authenticate yourself to GCP
terraform plan # check what Terraform is going to do
terraform apply # apply the changes

```

The next snippet demonstrates a typical [job configuration](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/), utilizing a *node selector* to designate the training node pool for the job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: "build-model"
spec:
  template:
    spec:
      containers:
      - name: "build-model"
        image: "python:3.7.3-alpine3.9"
        command: ["echo", "train my model"]
        resources:
          requests:
            cpu: 24
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-nodepool: "training-np"
  backoffLimit: 0
```

After deploying this job to our new GKE cluster, Kubernetes recognizes the need to create a new node in the training node pool due to the initial zero node count. It then runs the job on this new node. Ideally, after job completion, Kubernetes should reduce the node count back to zero. In GKE, this process generally takes 20 to 30 minutes. However, sometimes nodes remain in the pool because Kubernetes reallocates system services to our training nodes. To prevent this, we must incorporate a *taint* into our node pool configuration:

```hcl
...
  node_config {
    ...
    taint {
      key    = "special"
      value  = "strong-cpu"
      effect = "NO_SCHEDULE"
    }
  }
...
```

A taint ensures that Kubernetes assigns workloads only to nodes that tolerate that specific taint. Consequently, we also add a corresponding toleration to our job configuration:

```yaml
...
      tolerations:
      - key: "special"
        operator: "Equal"
        value: "strong-cpu"
        effect: "NoSchedule"
...
```

Implementing these changes guarantees reliable autoscaling and satisfies our management by ensuring we only pay for resources we actively use.
