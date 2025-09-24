# Dots Game Project

Fun pen-and-paper game from my school days. Mostly serves as proof of concept
for Uchen.ML in browser.

## Game Rules

Players take turns placing dots on a grid. The objective is to enclose areas (polygons) by connecting your dots, thereby claiming territory and scoring points. Key rules:

- **Turns:** Players alternate turns, placing one dot per turn.
- **Polygons:** When a player forms a closed loop (cycle) with their dots, the enclosed area is filled and claimed.
- **Scoring:** Players earn points for each cell enclosed by their polygons.
- **Expansion:** New polygons can expand existing ones, merging claimed areas.
- **Endgame:** The game ends when no more moves are possible or a win condition is met.

## Project Structure & Modules

- `src/` — React frontend (TypeScript). Basically, one big SVG.
- `wasm/` — Wasm wrapper for C++ game logic
- `game-cpp/` — Standalone C++ logic and tests, separated so can be built/tested independently and used for training.
- `uchen-core/` — Core ML/utility library

## Build & Run Instructions

### Prerequisites

- Node.js (for frontend)
- Bazel (for C++/WASM builds)
- C++ toolchain (system default recommended)

### 1. Build and Run the Frontend

```sh
# Install dependencies
npm install

# Start the development server
npm run dev
# Open http://localhost:5173 in your browser
```

### 2. Build the WASM Module

```sh
# From the project root or wasm/ directory
bazel build //wasm/src:main.wasm
# Output will be in wasm/bazel-bin/src/main.wasm
```

### 3. Run C++ Tests

```sh
# From the game-cpp/ directory
cd game-cpp
bazel test //test:game.test
```

## Notes

- Production build and deployment are manual. You'll have to figure it out...
