#!/bin/bash

SEED=42
BATCH_SIZE=16
EPOCHS=10
LR=1e-4
SYMBOLS="y"
WRITE_MODE="a"
MODEL_TYPES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_types)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                MODEL_TYPES+=("$1")
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

if [[ ${#MODEL_TYPES[@]} -eq 0 ]]; then
    echo "Error: You must provide at least one --model_type"
    echo "Allowed: enc, dec, enc-dec, qwen"
    exit 1
fi

ENC_DEC_MODELS=(
    "google-t5/t5-small"
    "google-t5/t5-base"
    "google-t5/t5-large"
    "google/flan-t5-small"
    "google/flan-t5-base"
    "google/flan-t5-large"
    "google/t5gemma-s-s-prefixlm"
    "google/t5gemma-2b-2b-ul2"
    "google/t5gemma-b-b-ul2"
)

DEC_MODELS=(
    "google/gemma-3-270m"
    "google/gemma-3-4b-it"
    "google/gemma-2-9b-it"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "mistralai/Mistral-7B-v0.2"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "EleutherAI/gpt-neo-2.7B"
)

ENC_MODELS=(
    "distilbert/distilroberta-base"
    "distilbert/distilbert-base-uncased"
    "microsoft/deberta-v3-base"
)

QWEN_MODELS=(
    "Qwen/Qwen2-Audio-7B-Instruct"
    "Qwen/Qwen2-Audio-7B"
)

FIRST_MODEL=true


for TYPE in "${MODEL_TYPES[@]}"; do

    case "$TYPE" in
        enc-dec)
            MODELS=("${ENC_DEC_MODELS[@]}")
            ;;
        dec)
            MODELS=("${DEC_MODELS[@]}")
            ;;
        enc)
            MODELS=("${ENC_MODELS[@]}")
            ;;
        qwen)
            MODELS=("${QWEN_MODELS[@]}")
            ;;
        *)
            echo "Unknown model type: $TYPE"
            exit 1
            ;;
    esac

    echo "============================================="
    echo "    $TYPE architectures training starts"
    echo "============================================="

    for MODEL in "${MODELS[@]}"; do

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

done