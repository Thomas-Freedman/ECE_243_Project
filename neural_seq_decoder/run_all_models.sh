#!/bin/bash
# Script to run all three models (baseline, advanced, hybrid) in parallel
# Each model runs in its own screen session for background execution

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Neural Decoder Training - All Models${NC}"
echo -e "${GREEN}========================================${NC}"

# Create results directories
echo -e "\n${BLUE}Creating output directories...${NC}"
mkdir -p ~/results/baseline
mkdir -p ~/results/advanced
mkdir -p ~/results/hybrid

echo -e "${GREEN}✓ Output directories created${NC}"

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo -e "\n${BLUE}Screen not installed. Installing...${NC}"
    sudo apt-get update && sudo apt-get install -y screen
fi

# Kill any existing screen sessions with these names
screen -X -S baseline quit 2>/dev/null || true
screen -X -S advanced quit 2>/dev/null || true
screen -X -S hybrid quit 2>/dev/null || true

echo -e "\n${BLUE}Starting training sessions...${NC}"

# Start baseline training in screen session
screen -dmS baseline bash -c "cd ~/neural_seq_decoder/src && PYTHONPATH=~/neural_seq_decoder/src:$PYTHONPATH python3 -m neural_decoder.main decoder=baseline datasetPath=~/competitionData/ptDecoder_ctc outputDir=~/results/baseline; exec bash"
echo -e "${GREEN}✓ Baseline training started in screen session 'baseline'${NC}"

# Start advanced training in screen session
screen -dmS advanced bash -c "cd ~/neural_seq_decoder/src && PYTHONPATH=~/neural_seq_decoder/src:$PYTHONPATH python3 -m neural_decoder.main decoder=advanced datasetPath=~/competitionData/ptDecoder_ctc outputDir=~/results/advanced; exec bash"
echo -e "${GREEN}✓ Advanced training started in screen session 'advanced'${NC}"

# Start hybrid training in screen session
screen -dmS hybrid bash -c "cd ~/neural_seq_decoder/src && PYTHONPATH=~/neural_seq_decoder/src:$PYTHONPATH python3 -m neural_decoder.main decoder=hybrid datasetPath=~/competitionData/ptDecoder_ctc outputDir=~/results/hybrid; exec bash"
echo -e "${GREEN}✓ Hybrid training started in screen session 'hybrid'${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All training sessions launched!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${BLUE}Useful commands:${NC}"
echo -e "  View all screen sessions:    ${GREEN}screen -ls${NC}"
echo -e "  Attach to baseline training: ${GREEN}screen -r baseline${NC}"
echo -e "  Attach to advanced training: ${GREEN}screen -r advanced${NC}"
echo -e "  Attach to hybrid training:   ${GREEN}screen -r hybrid${NC}"
echo -e "  Detach from screen:          ${GREEN}Ctrl+A then D${NC}"
echo -e "  Kill a session:              ${GREEN}screen -X -S <name> quit${NC}"
echo -e "\n${BLUE}Results will be saved to:${NC}"
echo -e "  Baseline: ${GREEN}~/results/baseline${NC}"
echo -e "  Advanced: ${GREEN}~/results/advanced${NC}"
echo -e "  Hybrid:   ${GREEN}~/results/hybrid${NC}"

echo -e "\n${BLUE}Monitor progress:${NC}"
echo -e "  ${GREEN}watch -n 5 'screen -ls'${NC}  # Check running sessions"
echo -e "  ${GREEN}ls -lh ~/results/*/modelWeights.pt${NC}  # Check for saved models"

echo ""
