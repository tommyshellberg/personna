#!/bin/bash
# Setup Qdrant vector database using Docker
# Usage: ./scripts/setup_qdrant.sh [start|stop|status|restart]

set -e

CONTAINER_NAME="qdrant"
IMAGE="qdrant/qdrant"
REST_PORT=6333
GRPC_PORT=6334
VOLUME_NAME="qdrant_storage"

usage() {
    echo "Usage: $0 [start|stop|status|restart]"
    echo ""
    echo "Commands:"
    echo "  start   - Start Qdrant container (creates if not exists)"
    echo "  stop    - Stop Qdrant container"
    echo "  status  - Check if Qdrant is running"
    echo "  restart - Restart Qdrant container"
    exit 1
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "Error: Docker daemon is not running"
        exit 1
    fi
}

start_qdrant() {
    check_docker

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        # Container exists, check if running
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Qdrant is already running"
            echo "REST API: http://localhost:${REST_PORT}"
            echo "gRPC: localhost:${GRPC_PORT}"
            return 0
        else
            # Container exists but stopped, start it
            echo "Starting existing Qdrant container..."
            docker start "${CONTAINER_NAME}"
        fi
    else
        # Create new container
        echo "Creating and starting Qdrant container..."
        docker run -d \
            -p ${REST_PORT}:6333 \
            -p ${GRPC_PORT}:6334 \
            -v ${VOLUME_NAME}:/qdrant/storage \
            --name ${CONTAINER_NAME} \
            ${IMAGE}
    fi

    # Wait for Qdrant to be ready
    echo "Waiting for Qdrant to be ready..."
    for i in {1..30}; do
        if curl -s "http://localhost:${REST_PORT}/readyz" > /dev/null 2>&1; then
            echo "Qdrant is ready!"
            echo "REST API: http://localhost:${REST_PORT}"
            echo "gRPC: localhost:${GRPC_PORT}"
            echo "Dashboard: http://localhost:${REST_PORT}/dashboard"
            return 0
        fi
        sleep 1
    done

    echo "Warning: Qdrant did not respond within 30 seconds"
    return 1
}

stop_qdrant() {
    check_docker

    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping Qdrant container..."
        docker stop "${CONTAINER_NAME}"
        echo "Qdrant stopped"
    else
        echo "Qdrant is not running"
    fi
}

status_qdrant() {
    check_docker

    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Qdrant is running"
        echo "REST API: http://localhost:${REST_PORT}"
        echo "gRPC: localhost:${GRPC_PORT}"

        # Check if API is responding
        if curl -s "http://localhost:${REST_PORT}/readyz" > /dev/null 2>&1; then
            echo "Status: Healthy"
        else
            echo "Status: Container running but API not responding"
        fi
    else
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Qdrant container exists but is stopped"
            echo "Run '$0 start' to start it"
        else
            echo "Qdrant container does not exist"
            echo "Run '$0 start' to create and start it"
        fi
    fi
}

restart_qdrant() {
    stop_qdrant
    start_qdrant
}

# Main
case "${1:-}" in
    start)
        start_qdrant
        ;;
    stop)
        stop_qdrant
        ;;
    status)
        status_qdrant
        ;;
    restart)
        restart_qdrant
        ;;
    *)
        usage
        ;;
esac
