package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/rs/cors"
)

var jwtKey = []byte("my_secret_key")

// User credentials map (manual management)
var users = map[string]string{
	"admin": "securepassword",
}

// Job represents a music generation task
type Job struct {
	Prompt     string    `json:"prompt"`
	Filename   string    `json:"filename"`
	Duration   int       `json:"duration,omitempty"` // Duration in seconds
	Status     string    `json:"status"`             // "queued", "processing", "completed", "failed"
	StartedAt  time.Time `json:"started_at,omitempty"`
	FinishedAt time.Time `json:"finished_at,omitempty"`
}

// Default durations based on complexity of prompt
var complexityDurations = map[string]int{
	"simple":  30, // 30 seconds for simple prompts
	"medium":  45, // 45 seconds for medium complexity
	"complex": 60, // 60 seconds for complex prompts
}

// Job queue and mutex
var jobQueue []Job
var jobMutex sync.Mutex

// API configuration
const (
	// Adjust this to point to your Python server's address
	// For local development, this might be "http://localhost:8001"
	// For Docker setup, use the service name like "http://musicgen-api:8001"
	MusicGenAPIURL = "http://musicgen-api:8001"
)

func main() {
	mux := http.NewServeMux()

	// Auth and job routes
	mux.HandleFunc("/login", loginHandler)
	mux.HandleFunc("/generate", generateHandler)
	mux.HandleFunc("/queue-status", queueStatusHandler)

	// Serve WAV files from NAS mount
	mux.Handle("/files/", http.StripPrefix("/files/", http.FileServer(http.Dir("/mnt/files"))))

	// Enhanced CORS for frontend - supports multiple common development ports
	handler := cors.New(cors.Options{
		AllowedOrigins: []string{
			"*", // Temporary - allow all origins for testing
		},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type", "Authorization"},
		AllowCredentials: true,
	}).Handler(mux)

	fmt.Println("Server is running at http://localhost:8000")
	fmt.Println("CORS enabled for ports: 3000, 5173, 8080")
	fmt.Println("Available endpoints:")
	fmt.Println("  POST /login - User authentication")
	fmt.Println("  POST /generate - Music generation")
	fmt.Println("  GET /queue-status - Job queue status")
	fmt.Println("  GET /files/* - Serve generated WAV files")

	log.Fatal(http.ListenAndServe(":8000", handler))
}

// loginHandler issues a JWT token for valid credentials
// loginHandler handles user authentication and JWT token generation
func loginHandler(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers explicitly for this endpoint
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Content-Type", "application/json")

	// Handle preflight OPTIONS request
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Method not allowed. Use POST.",
		})
		return
	}

	// Read and parse request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Failed to read request body",
		})
		return
	}
	defer r.Body.Close()

	// Parse JSON credentials
	var credentials struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}

	if err := json.Unmarshal(body, &credentials); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Invalid JSON format",
		})
		return
	}

	// Validate required fields
	if credentials.Username == "" || credentials.Password == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Username and password are required",
		})
		return
	}

	// Check credentials against user store
	storedPassword, userExists := users[credentials.Username]
	if !userExists || storedPassword != credentials.Password {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Invalid username or password",
		})
		return
	}

	// Create JWT token
	now := time.Now()
	claims := jwt.MapClaims{
		"username": credentials.Username,
		"iat":      now.Unix(),
		"exp":      now.Add(24 * time.Hour).Unix(), // 24 hour expiration
		"iss":      "music-gen-server",
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(jwtKey)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Failed to generate authentication token",
		})
		return
	}

	// Success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"token":   tokenString,
		"user":    credentials.Username,
		"expires": now.Add(24 * time.Hour).Unix(),
	})
}

// Helper to determine duration based on prompt complexity
func determineDuration(prompt string) int {
	// Default duration is medium (45 seconds)
	duration := complexityDurations["medium"]

	// Count words in prompt
	wordCount := len(strings.Fields(prompt))

	// Very simple heuristic - could be made more sophisticated
	if wordCount < 5 {
		duration = complexityDurations["simple"]
	} else if wordCount > 10 {
		duration = complexityDurations["complex"]
	}

	return duration
}

// generateHandler adds a job to the queue and starts processing
func generateHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Generate request from %s: %s %s", r.RemoteAddr, r.Method, r.URL.Path)

	// Validate JWT token
	tokenString := r.Header.Get("Authorization")
	if tokenString == "" {
		log.Printf("Missing Authorization header")
		http.Error(w, "Missing Authorization header", http.StatusUnauthorized)
		return
	}

	tokenString = strings.TrimPrefix(tokenString, "Bearer ")
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jwtKey, nil
	})

	if err != nil || !token.Valid {
		log.Printf("Invalid token: %v", err)
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	// Parse request body
	var data map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		log.Printf("Error decoding request body: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	prompt, ok := data["prompt"].(string)
	if !ok || prompt == "" {
		log.Printf("Missing or invalid prompt")
		http.Error(w, "Missing or invalid prompt", http.StatusBadRequest)
		return
	}

	filename, ok := data["filename"].(string)
	if !ok || filename == "" {
		log.Printf("Missing or invalid filename")
		http.Error(w, "Missing or invalid filename", http.StatusBadRequest)
		return
	}

	log.Printf("New generation request - Prompt: %s, Filename: %s", prompt, filename)

	// Get duration if provided, otherwise determine based on prompt
	var duration int
	if durationFloat, ok := data["duration"].(float64); ok {
		duration = int(durationFloat)
	} else {
		duration = determineDuration(prompt)
	}

	// Cap duration to a reasonable value
	if duration > 60 {
		duration = 60
	}

	log.Printf("Determined duration: %d seconds", duration)

	// Create job
	job := Job{
		Prompt:   prompt,
		Filename: filename,
		Duration: duration,
		Status:   "queued",
	}

	// Add job to queue
	jobMutex.Lock()
	jobQueue = append(jobQueue, job)
	queueLength := len(jobQueue)
	jobMutex.Unlock()

	log.Printf("Job added to queue. Queue length: %d", queueLength)

	// Process job in background
	go runJob(job)

	// Send response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"message": fmt.Sprintf("Job queued! Will generate %d seconds of audio.", duration),
	})
}

// runJob starts and tracks a job's lifecycle
func runJob(job Job) {
	log.Printf("Starting job processing for: %s", job.Filename)

	// Update job status to "processing"
	jobMutex.Lock()
	var jobIndex int
	for i := range jobQueue {
		if jobQueue[i].Filename == job.Filename && jobQueue[i].Prompt == job.Prompt {
			jobQueue[i].Status = "processing"
			jobQueue[i].StartedAt = time.Now()
			jobIndex = i
			break
		}
	}
	jobMutex.Unlock()

	// Prepare request payload
	payload := map[string]interface{}{
		"prompt":   job.Prompt,
		"outpath":  job.Filename + ".wav",
		"duration": job.Duration,
	}

	payloadBytes, _ := json.Marshal(payload)

	log.Printf("📤 Sending JSON to %s/generate: %s", MusicGenAPIURL, string(payloadBytes))

	// Send request to Python server
	resp, err := http.Post(MusicGenAPIURL+"/generate", "application/json", bytes.NewBuffer(payloadBytes))
	if err != nil {
		log.Printf("❌ Failed to call Python server: %v", err)
		jobMutex.Lock()
		jobQueue[jobIndex].Status = "failed"
		jobQueue[jobIndex].FinishedAt = time.Now()
		jobMutex.Unlock()
		return
	}
	defer resp.Body.Close()

	// Process response
	body, _ := io.ReadAll(resp.Body)
	log.Printf("Python server response (%d): %s", resp.StatusCode, body)

	// Update job status based on response
	jobMutex.Lock()
	if resp.StatusCode == http.StatusOK {
		jobQueue[jobIndex].Status = "completed"
		log.Printf("✅ Job completed successfully: %s", job.Filename)
	} else {
		jobQueue[jobIndex].Status = "failed"
		log.Printf("❌ Job failed: %s", job.Filename)
	}
	jobQueue[jobIndex].FinishedAt = time.Now()
	jobMutex.Unlock()
}

// queueStatusHandler returns all jobs
func queueStatusHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Queue status request from %s", r.RemoteAddr)

	jobMutex.Lock()
	defer jobMutex.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"jobs":  jobQueue,
		"count": len(jobQueue),
	})
}
