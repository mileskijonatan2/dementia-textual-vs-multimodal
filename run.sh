#!/bin/bash

SEED=42
BATCH_SIZE=16
EPOCHS=10
LR=1e-4
SYMBOLS="y"
WRITE_MODE="a"
MODEL_NAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_names)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                MODEL_NAMES+=("$1")
                shift
            done
            ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --lr)
            LR="$2"; shift 2 ;;
        --symbols)
            SYMBOLS="$2"; shift 2 ;;
        --write_mode)
            WRITE_MODE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if [[ ${#MODEL_NAMES[@]} -eq 0 ]]; then
    echo "Error: You must provide at least one --model_name"
    echo "Valid model names:"
    echo "google-t5/t5-small"
    echo "google-t5/t5-base"
    echo "google-t5/t5-large"
    echo "google/flan-t5-small"
    echo "google/flan-t5-base"
    echo "google/flan-t5-large"
    echo "google/t5gemma-s-s-prefixlm"
    echo "google/t5gemma-2b-2b-ul2"
    echo "google/t5gemma-b-b-ul2"
    echo "google/gemma-3-270m"
    echo "google/gemma-3-4b-it"
    echo "google/gemma-2-9b-it"
    echo "mistralai/Mistral-7B-Instruct-v0.2"
    echo "mistralai/Mistral-7B-v0.2"
    echo "meta-llama/Llama-3.2-1B-Instruct"
    echo "meta-llama/Llama-3.2-3B-Instruct"
    echo "meta-llama/Llama-3.1-8B-Instruct"
    echo "EleutherAI/gpt-neo-2.7B"
    echo "distilbert/distilroberta-base"
    echo "distilbert/distilbert-base-uncased"
    echo "microsoft/deberta-v3-base"
    echo "Qwen/Qwen2-Audio-7B-Instruct"
    echo "Qwen/Qwen2-Audio-7B"
    exit 1
fi

FIRST_MODEL=true

for MODEL in "${MODEL_NAMES[@]}"; do
    echo "============================================="
    echo "    $MODEL architectures training starts"
    echo "============================================="

    if [ "$FIRST_MODEL" = true ]; then
        CURRENT_WRITE_MODE="$WRITE_MODE"
        FIRST_MODEL=false
    else
        CURRENT_WRITE_MODE="a"
    fi

    CMD="python train.py \
        --model $MODEL \
        --seed $SEED \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --symbols $SYMBOLS \
        --write_mode $CURRENT_WRITE_MODE"

    echo "Running command:"
    echo "$CMD"
    echo ""

    $CMD

    echo ""
    echo "++++++++++++++++++++++++++++++++++++"
    echo ""

done