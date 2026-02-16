#!/usr/bin/env ruby
# frozen_string_literal: true

# Deploy tensors to junkpile
# Usage: ruby scripts/deploy.rb

require "open3"

REMOTE = "chi@junkpile"
REMOTE_DIR = "/opt/tensors/app"
LOCAL_DIR = File.expand_path("..", __dir__)

def run(cmd, desc: nil)
  puts "==> #{desc}" if desc
  puts "    $ #{cmd}" if ENV["DEBUG"]

  stdout, stderr, status = Open3.capture3(cmd)

  unless status.success?
    puts "ERROR: #{stderr}" unless stderr.empty?
    puts stdout unless stdout.empty?
    exit 1
  end

  stdout
end

def ssh(cmd)
  run(%(ssh #{REMOTE} "#{cmd}"))
end

puts "==> Syncing Python code to junkpile..."
run(<<~CMD.gsub("\n", " ").strip)
  rsync -av --delete
  --exclude='.git'
  --exclude='__pycache__'
  --exclude='.venv'
  --exclude='node_modules'
  --exclude='.ruff_cache'
  --exclude='.mypy_cache'
  --exclude='.pytest_cache'
  --exclude='*.egg-info'
  --rsync-path="sudo rsync"
  #{LOCAL_DIR}/tensors/ #{REMOTE}:#{REMOTE_DIR}/tensors/
CMD

puts ""
puts "==> Fixing permissions..."
ssh("sudo chown -R tensors:tensors #{REMOTE_DIR} && sudo chmod -R g+w #{REMOTE_DIR}")

puts ""
puts "==> Restarting tensors service..."
ssh("sudo systemctl restart tensors")

puts ""
puts "==> Waiting for tensors to start..."
sleep 2

puts ""
puts "==> Verifying tensors service..."
status = ssh("systemctl is-active tensors").strip

if status == "active"
  puts "✓ tensors service running"
else
  puts "✗ tensors service not running (status: #{status})"
  puts ssh("journalctl -u tensors -n 10 --no-pager")
  exit 1
end

puts ""
puts "==> Checking API health..."
response = ssh("curl -sf http://127.0.0.1:51200/status || echo 'FAILED'").strip

if response.include?("FAILED")
  puts "✗ API health check failed"
  exit 1
else
  puts "✓ API responding"
end

puts ""
puts "==> Deploy complete!"
puts "    API: https://tensors-api.saiden.dev"
