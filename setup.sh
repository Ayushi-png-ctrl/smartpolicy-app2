#!/bin/bash
mkdir -p ~/.streamlit

echo "[theme]
primaryColor='#1E3D58'
backgroundColor='#FFFFFF'
secondaryBackgroundColor='#F0F2F6'
textColor='#31333F'
font='sans serif'

[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
