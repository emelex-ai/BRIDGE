provider "google" {
  project = var.project_id
  region  = var.region
}

# Reference existing GCS bucket for datasets and models
resource "google_storage_bucket" "ml_storage" {
  name     = "bridge-ml-training" 
  location = var.region
  force_destroy = false  # Protect existing bucket from accidental deletion
  
  uniform_bucket_level_access = false
  
  versioning {
    enabled = false
  }
  
  # Prevent Terraform from destroying this existing resource
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_service_account" "training_service_account" {
  account_id   = "bridge-training-sa"
  display_name = "ML Training Service Account"
  description  = "Service account for ML training jobs"
}

# Grant storage permissions to the service account
resource "google_storage_bucket_iam_member" "storage_object_admin" {
  bucket = google_storage_bucket.ml_storage.name
  role   = "roles/storage.objectAdmin"  
  member = "serviceAccount:${google_service_account.training_service_account.email}"
}

# Allow Cloud Run to use the service account
resource "google_project_iam_member" "service_account_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.training_service_account.email}"
}

# Set up Artifact Registry
resource "google_artifact_registry_repository" "ml_containers" {
  location      = var.region
  repository_id = "ml-training-images"
  description   = "Docker repository for ML training containers"
  format        = "DOCKER"
}

# Reference artifact registry image
data "google_artifact_registry_docker_image" "pretraining" {
  repository_id = "ml-training-images"
  location = var.region
  image_name = "pretraining:latest"
}

# Create Cloud Run Job
resource "google_cloud_run_v2_job" "pretraining_job" {
  name = "pretraining-job"
  location = var.region
  
  template {
    task_count = 50
    template {
      timeout = "36000s" # 10 hours
      service_account = google_service_account.training_service_account.email
      containers {
        image = data.google_artifact_registry_docker_image.pretraining.self_link
        
        resources {
          limits = {
            memory = "2Gi"
            cpu    = "1"
          }
        }
        env {
          name  = "BUCKET_NAME"
          value = google_storage_bucket.ml_storage.name
        }
      }

    }
  }

}


# Grant service account permission to pull images
resource "google_artifact_registry_repository_iam_member" "registry_reader" {
  location   = google_artifact_registry_repository.ml_containers.location
  repository = google_artifact_registry_repository.ml_containers.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.training_service_account.email}"
}

# Define variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "bridge-457501"
}

variable "region" {
  description = "GCP Region"
  default     = "us-central1"
  type        = string
}

# Output important values
output "gcs_bucket" {
  value = google_storage_bucket.ml_storage.name
}

output "service_account" {
  value = google_service_account.training_service_account.email
}

output "container_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_containers.name}"
}