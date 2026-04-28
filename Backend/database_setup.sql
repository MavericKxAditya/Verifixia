-- Database Setup for Deepfake Detection App

-- Create Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create Detection Logs table
CREATE TABLE IF NOT EXISTS detection_logs (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    prediction VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER REFERENCES users(id)
);

-- Create Index on timestamp for faster queries
CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp ON detection_logs(timestamp);
