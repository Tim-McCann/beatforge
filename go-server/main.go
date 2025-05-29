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
	Duration   int       `json:"duration,omitempty"`
	Status     string    `json:"status"`
	StartedAt  time.Time `json:"started_at,omitempty"`
	FinishedAt time.Time `json:"finished_at,omitempty"`
}

// Default durations based on complexity of prompt
var complexityDurations = map[string]int{
	"simple":  30,
	"medium":  45,
	"complex": 60,
}

// Initialize jobQueue as empty slice instead of nil
var jobQueue = []Job{}
var jobMutex sync.Mutex

const MusicGenAPIURL = "http://musicgen-api:8001"

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/login", loginHandler)
	mux.HandleFunc("/generate", generateHandler)
	mux.HandleFunc("/queue-status", queueStatusHandler)
	mux.Handle("/files/", http.StripPrefix("/files/", http.FileServer(http.Dir("/mnt/files"))))

	handler := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type", "Authorization"},
		AllowCredentials: true,
	}).Handler(mux)

	fmt.Println("Server running at http://localhost:8000")
	log.Fatal(http.ListenAndServe(":8000", handler))
}

func loginHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "Method not allowed. Use POST."})
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to read request body"})
		return
	}
	defer r.Body.Close()

	var credentials struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}

	if err := json.Unmarshal(body, &credentials); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON format"})
		return
	}

	if credentials.Username == "" || credentials.Password == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Username and password are required"})
		return
	}

	storedPassword, userExists := users[credentials.Username]
	if !userExists || storedPassword != credentials.Password {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid username or password"})
		return
	}

	now := time.Now()
	claims := jwt.MapClaims{
		"username": credentials.Username,
		"iat":      now.Unix(),
		"exp":      now.Add(24 * time.Hour).Unix(),
		"iss":      "music-gen-server",
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(jwtKey)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to generate token"})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"token":   tokenString,
		"user":    credentials.Username,
		"expires": now.Add(24 * time.Hour).Unix(),
	})
}

func determineDuration(prompt string) int {
	wordCount := len(strings.Fields(prompt))
	if wordCount < 5 {
		return complexityDurations["simple"]
	} else if wordCount > 10 {
		return complexityDurations["complex"]
	}
	return complexityDurations["medium"]
}

func generateHandler(w http.ResponseWriter, r *http.Request) {
	tokenString := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		return jwtKey, nil
	})
	if err != nil || !token.Valid {
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	var data map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	prompt, ok := data["prompt"].(string)
	if !ok || prompt == "" {
		http.Error(w, "Missing or invalid prompt", http.StatusBadRequest)
		return
	}
	filename, ok := data["filename"].(string)
	if !ok || filename == "" {
		http.Error(w, "Missing or invalid filename", http.StatusBadRequest)
		return
	}

	duration := determineDuration(prompt)
	if d, ok := data["duration"].(float64); ok {
		duration = int(d)
		if duration > 60 {
			duration = 60
		}
	}

	job := Job{Prompt: prompt, Filename: filename, Duration: duration, Status: "queued"}

	jobMutex.Lock()
	jobQueue = append(jobQueue, job)
	jobMutex.Unlock()

	go runJob(job)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"message": fmt.Sprintf("Job queued! Will generate %d seconds of audio.", duration),
	})
}

func runJob(job Job) {
	jobMutex.Lock()
	for i := range jobQueue {
		if jobQueue[i].Filename == job.Filename && jobQueue[i].Prompt == job.Prompt {
			jobQueue[i].Status = "processing"
			jobQueue[i].StartedAt = time.Now()
			break
		}
	}
	jobMutex.Unlock()

	payload := map[string]interface{}{
		"prompt":   job.Prompt,
		"outpath":  job.Filename + ".wav",
		"duration": job.Duration,
	}
	payloadBytes, _ := json.Marshal(payload)

	resp, err := http.Post(MusicGenAPIURL+"/generate", "application/json", bytes.NewBuffer(payloadBytes))
	jobMutex.Lock()
	for i := range jobQueue {
		if jobQueue[i].Filename == job.Filename && jobQueue[i].Prompt == job.Prompt {
			if err != nil || resp.StatusCode != http.StatusOK {
				jobQueue[i].Status = "failed"
			} else {
				jobQueue[i].Status = "completed"
				jobQueue[i].FinishedAt = time.Now()
			}
			break
		}
	}
	jobMutex.Unlock()
	if resp != nil {
		resp.Body.Close()
	}
}

func queueStatusHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	jobMutex.Lock()
	defer jobMutex.Unlock()
	
	// Ensure jobQueue is never nil
	if jobQueue == nil {
		jobQueue = []Job{}
	}
	
	response := map[string]interface{}{
		"count": len(jobQueue),
		"jobs":  jobQueue,
	}
	
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding JSON response: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to encode response"})
		return
	}
}