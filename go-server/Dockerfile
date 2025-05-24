FROM golang:1.20-alpine

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git

# Copy the Go module files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy the source code
COPY main.go ./

# Build the application
RUN go build -o musicgen-server

# Create directory for mounted files
RUN mkdir -p /mnt/files

# Expose port
EXPOSE 8000

# Run the server
CMD ["./musicgen-server"]