# CLAUDE.md - GigaLearnCPP Developer Guide for AI Assistants

## Project Overview

**GigaLearnCPP** is a high-performance C++ machine learning framework for training Rocket League bots using Proximal Policy Optimization (PPO). This is a leaked version of what was originally a private library.

**Performance Claims**:
- ~2x faster than RLGymPPO-CPP for data collection
- ~10x faster than RLGym-PPO (Python) for data collection
- Single-process monolithic inference engine

**Key Technologies**:
- **Language**: C++20
- **Build System**: CMake 3.8+
- **ML Framework**: PyTorch C++ (LibTorch)
- **Physics Engine**: RocketSim (custom, based on Bullet Physics 3.24)
- **Algorithm**: Proximal Policy Optimization (PPO)

---

## Repository Structure

```
GigaLearnCPP-Leak/
├── GigaLearnCPP/              # Main ML framework library (SHARED)
│   ├── src/
│   │   ├── public/            # Public API headers
│   │   │   └── GigaLearnCPP/
│   │   │       ├── Learner.h/cpp          # Main training orchestrator
│   │   │       ├── LearnerConfig.h        # Configuration structs
│   │   │       ├── Framework.h            # Common imports/macros
│   │   │       ├── PPO/                   # PPO configs
│   │   │       └── Util/                  # Utilities (InferUnit, Report, etc.)
│   │   └── private/           # Implementation details
│   │       └── GigaLearnCPP/
│   │           ├── FrameworkTorch.h       # PyTorch integration
│   │           ├── PPO/                   # PPOLearner, ExperienceBuffer, GAE
│   │           └── Util/                  # Models, optimizers
│   ├── RLGymCPP/              # RL Gym environment (STATIC)
│   │   └── src/RLGymCPP/
│   │       ├── EnvSet/                    # Parallel environment manager
│   │       ├── Gamestates/                # GameState, Player
│   │       ├── Rewards/                   # Reward interface + built-ins
│   │       ├── OBSBuilders/               # Observation builders
│   │       ├── ActionParsers/             # Action space definitions
│   │       ├── StateSetters/              # Initial state generators
│   │       ├── TerminalConditions/        # Episode termination logic
│   │       └── RocketSim/                 # Physics simulator
│   ├── pybind11/              # Python bindings (submodule)
│   ├── libsrc/json/           # nlohmann/json library
│   └── python_scripts/        # WandB metrics receiver, visualization
│
├── RLBotCPP/                  # RLBot framework client (submodule, STATIC)
├── src/                       # Example bot implementation
│   ├── ExampleMain.cpp        # Complete training example ⭐
│   └── RLBotClient.h/cpp      # RLBot integration wrapper
├── rlbot/                     # RLBot configuration files
├── tools/                     # Utilities (checkpoint converter)
└── CMakeLists.txt             # Root build config
```

---

## Core Components

### 1. Learner (Main Entry Point)

**Location**: `/GigaLearnCPP/src/public/GigaLearnCPP/Learner.h`

The `Learner` class orchestrates the entire training process.

**Responsibilities**:
- Manages environment set (`EnvSet`)
- Runs PPO training loop (`PPOLearner`)
- Handles checkpoint saving/loading
- Integrates metrics reporting (WandB)
- Manages skill tracking and policy versioning
- Provides visualization support

**Key Methods**:
- `Learner(EnvCreateFunc, LearnerConfig)` - Constructor
- `Start()` - Begin training (blocking)
- `SaveCheckpoint(path)` - Manual save
- `LoadCheckpoint(path)` - Load from disk

### 2. Configuration System

**Location**: `/GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h`

Configuration is done via plain structs with sensible defaults.

**Key Config Structs**:
- `LearnerConfig` - Top-level configuration
- `PPOLearnerConfig` - PPO hyperparameters
- `ModelConfig` - Neural network architecture
- `SkillTrackerConfig` - ELO-based skill tracking

**Important Config Fields**:
```cpp
LearnerConfig cfg = {};
cfg.numGames = 256;                    // Parallel environments
cfg.timestepsPerIteration = 50000;     // Steps before PPO update
cfg.ppo.batchSize = 50000;             // PPO batch size
cfg.ppo.epochs = 30;                   // PPO epochs per iteration
cfg.ppo.policy.layerSizes = {256, 256, 256};
cfg.ppo.critic.layerSizes = {256, 256, 256};
cfg.device = torch::kCUDA;             // Or kCPU, kAUTO
cfg.renderSendRate = 0.5f;             // Render every N seconds
cfg.metricsSendRate = 5;               // Metrics every N iterations
```

### 3. Environment System (RLGymCPP)

**Location**: `/GigaLearnCPP/RLGymCPP/`

**EnvSet** - Manages parallel environments with thread pool:
```cpp
EnvSet* envSet = new EnvSet(envCreateFunc, numGames);
envSet->Reset();
auto obs = envSet->Step(actions, rewards, dones);
```

**Plugin Interfaces** (user implements these):

#### Reward Interface
```cpp
class MyReward : public RLGC::Reward {
public:
    virtual float GetReward(const RLGC::Player& player,
                           const RLGC::GameState& state,
                           bool isFinal) override {
        // Access current state
        float speed = player.vel.Length();

        // Access previous state (temporal rewards)
        if (player.prev) {
            float prevSpeed = player.prev->vel.Length();
        }

        // Check events
        if (player.eventState.shot) {
            return 10.0f;
        }

        return 0.0f;
    }
};
```

#### Observation Builder
```cpp
class MyObsBuilder : public RLGC::ObsBuilder {
public:
    virtual RLGC::FList BuildObs(const RLGC::Player& player,
                                 const RLGC::GameState& state) override {
        RLGC::FList obs;
        obs.push_back(player.pos.x);
        obs.push_back(player.pos.y);
        // ... add more features
        return obs;
    }

    virtual int GetObsSize() override { return 107; }
};
```

#### Action Parser
```cpp
class MyActionParser : public RLGC::ActionParser {
public:
    virtual RLGC::Action ParseAction(int action,
                                     const RLGC::GameState& state) override {
        // Convert discrete index to controller input
        RLGC::Action result = {};
        result.throttle = actions[action].throttle;
        result.steer = actions[action].steer;
        // ...
        return result;
    }

    virtual int GetActionAmount() override { return 90; }

    // Optional: action masking
    virtual std::vector<float> GetActionMask() override {
        return std::vector<float>(90, 1.0f); // All actions valid
    }
};
```

#### State Setter
```cpp
class MyStateSetter : public RLGC::StateSetter {
public:
    virtual void ResetArena(RS::Arena* arena) override {
        // Reset to kickoff
        arena->ball->pos = {0, 0, 100};
        arena->ball->vel = {0, 0, 0};

        for (auto car : arena->GetCars()) {
            // Position cars, set boost, etc.
        }
    }
};
```

### 4. Inference System

**Location**: `/GigaLearnCPP/src/public/GigaLearnCPP/Util/InferUnit.h`

`InferUnit` is used for deploying trained models (e.g., in RLBot):

```cpp
InferUnit* inferUnit = new InferUnit(
    obsBuilder, obsSize, actionParser,
    sharedHeadConfig, policyConfig,
    "checkpoints/1000000/",  // Checkpoint path
    true  // Use GPU
);

// Single inference
int action = inferUnit->InferSingle(obs);

// Batch inference
auto actions = inferUnit->InferBatch(obsBatch);
```

### 5. GameState and Player

**Location**: `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates/`

**GameState** - Complete game state:
```cpp
struct GameState {
    std::vector<Player*> players;
    BallState ball;
    std::vector<bool> pads;        // Boost pad active states
    std::vector<float> padTimers;  // Boost pad respawn timers
    GameState* prev;               // Previous state (or nullptr)
};
```

**Player** - Enhanced player/car state:
```cpp
struct Player : public CarState {
    // Inherited from CarState (RocketSim):
    Vec pos, vel, angVel;
    RotMat rotMat;
    bool isOnGround, hasJumped, hasFlipped;
    float boost;
    // ... many more physics fields

    // GigaLearnCPP additions:
    Player* prev;                  // Previous player state
    Action prevAction;             // Last action taken
    EventState eventState;         // goal, save, shot, assist, demo, etc.
    bool ballTouched;              // Touched ball this step
    int carId;                     // Unique car identifier
};
```

**Event Tracking**:
```cpp
if (player.eventState.shot) { /* reward shooting */ }
if (player.eventState.goal) { /* reward scoring */ }
if (player.eventState.save) { /* reward saving */ }
if (player.eventState.demolish) { /* reward demo */ }
```

---

## Development Setup

### Prerequisites

1. **LibTorch** (PyTorch C++)
   - Download from: https://pytorch.org/get-started/locally/
   - Choose: C++/LibTorch, your platform, CUDA version (or CPU)
   - Extract to `/GigaLearnCPP/libtorch/` OR system path

2. **Python 3.x** with development headers
   - For embedding Python interpreter (WandB metrics)
   - Install: `python3-dev` (Linux) or include in Python install (Windows)

3. **CMake** 3.8 or higher

4. **Arena Collision Meshes** (`.cmf` files)
   - Required for RocketSim initialization
   - Dump from Rocket League using tools like RLArenaCollisionDumper
   - Place in project root or specify path in code

5. **Git** (for cloning submodules)

### Build Instructions

```bash
# Clone with submodules
git clone --recursive https://github.com/your-repo/GigaLearnCPP-Leak.git
cd GigaLearnCPP-Leak

# If already cloned without --recursive
git submodule update --init --recursive

# Create build directory
mkdir build && cd build

# Configure (adjust LibTorch path as needed)
# Windows (CUDA):
cmake .. -DCMAKE_PREFIX_PATH="C:/path/to/libtorch"

# Linux (CUDA):
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch

# CPU-only:
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch-cpu

# Build
cmake --build . --config Release

# Outputs:
# - build/GigaLearnBot (executable)
# - build/GigaLearnCPP.so/.dll (shared library)
# - build/python_scripts/ (metrics scripts)
```

### Troubleshooting Build Issues

**LibTorch not found**:
- Ensure `CMAKE_PREFIX_PATH` points to extracted LibTorch directory
- Verify `libtorch/share/cmake/Torch/TorchConfig.cmake` exists

**Python errors**:
- Install Python development headers: `sudo apt-get install python3-dev`
- Ensure Python is in PATH

**Missing collision meshes**:
- Code will error at runtime if `.cmf` files not found
- Dump using RLArenaCollisionDumper or similar tool
- Load in code: `Arena::Create(GameMode::SOCCAR, "soccar.cmf")`

**CUDA errors**:
- Verify CUDA version matches LibTorch build
- Check GPU drivers are up-to-date
- For CPU-only, download CPU version of LibTorch

---

## Key Conventions

### Namespace Conventions

- `GGL::` - GigaLearn framework namespace
- `RLGC::` - RLGymCPP environment namespace
- `RS::` - RocketSim physics namespace

### Naming Conventions

- **Classes**: PascalCase (`PPOLearner`, `EnvSet`, `InferUnit`)
- **Methods**: PascalCase (`GetReward`, `BuildObs`, `StepOptim`)
- **Variables**: camelCase (`obsSize`, `numActions`, `prevAction`)
- **Config Structs**: PascalCase + `Config` suffix (`LearnerConfig`)
- **Macros**: UPPER_SNAKE_CASE (`RG_NO_COPY`, `RG_ERR_CLOSE`)

### Code Organization Patterns

**Public/Private Split**:
- Public API: `/src/public/GigaLearnCPP/` - User-facing headers
- Private impl: `/src/private/GigaLearnCPP/` - Internal implementation

**Plugin Architecture**:
- Abstract base classes with virtual methods
- User implements custom behavior via inheritance
- Framework calls via polymorphism

**Configuration Objects**:
- Plain structs (not classes)
- Default-initialized fields
- Passed by value or const reference

**Previous State Access**:
- `GameState* prev` and `Player* prev` pointers
- Enables temporal reward functions
- Chain maintained by framework (don't modify)

**Memory Management**:
- Raw pointers for plugin interfaces (user manages lifetime)
- Manual `delete` in destructors (older C++ style)
- Plugins must live for duration of training

### Important Macros

```cpp
RG_NO_COPY(ClassName)        // Disable copy constructor/assignment
RG_ERR_CLOSE(msg)            // Log error and exit
RG_LOG(msg)                  // Log message
RG_CUDA_SUPPORT              // Defined if CUDA available
```

---

## Common Workflows

### 1. Setting Up a Training Run

**File**: `/src/ExampleMain.cpp` (reference implementation)

```cpp
#include <GigaLearnCPP/Framework.h>

// Step 1: Define environment creation function
RLGC::EnvCreateResult EnvCreateFunc(int index) {
    // Create arena
    auto arena = RS::Arena::Create(RS::GameMode::SOCCAR);

    // Add cars
    for (int i = 0; i < 6; i++) {
        auto car = arena->AddCar(i % 2 == 0 ? RS::Team::BLUE : RS::Team::ORANGE);
    }

    // Create reward function (weighted combination)
    std::vector<RLGC::WeightedReward> rewards = {
        {new RLGC::GoalReward(), 10.0f},
        {new RLGC::TouchReward(), 0.1f},
        {new RLGC::VelocityReward(), 0.01f}
    };
    auto reward = new RLGC::CombinedReward(rewards);

    // Wrap for competitive zero-sum
    reward = new RLGC::ZeroSumReward(reward);

    // Terminal conditions
    std::vector<RLGC::TerminalCondition*> termConds = {
        new RLGC::GoalScoreCondition(),
        new RLGC::NoTouchTimeoutCondition(225)
    };

    // Observation builder
    auto obsBuilder = new RLGC::DefaultObs();

    // Action parser
    auto actionParser = new RLGC::DiscreteAction();

    // State setter (initial state)
    auto stateSetter = new RLGC::RandomState();

    RLGC::EnvCreateResult result = {
        arena, reward, obsBuilder, actionParser,
        termConds, stateSetter
    };
    return result;
}

// Step 2: Configure learner
int main() {
    GGL::LearnerConfig cfg = {};

    // Environment settings
    cfg.numGames = 256;              // Parallel environments
    cfg.timestepsPerIteration = 50000;
    cfg.tickSkip = 8;                // Game ticks per action
    cfg.actionDelay = 0;             // Input lag simulation

    // PPO hyperparameters
    cfg.ppo.batchSize = 50000;
    cfg.ppo.miniBatchSize = 25000;
    cfg.ppo.epochs = 30;
    cfg.ppo.policyLR = 5e-5f;
    cfg.ppo.criticLR = 5e-5f;
    cfg.ppo.gamma = 0.99f;
    cfg.ppo.gaeLambda = 0.95f;
    cfg.ppo.entropyCoeff = 0.01f;

    // Model architecture
    cfg.ppo.sharedHead.layerSizes = {256, 256};  // Shared layers
    cfg.ppo.policy.layerSizes = {256};           // Policy head
    cfg.ppo.critic.layerSizes = {256};           // Critic head
    cfg.ppo.policy.activation = GGL::EActivation::RELU;
    cfg.ppo.policy.optimizer = GGL::EOptimizer::ADAM;

    // Device
    cfg.device = torch::kCUDA;  // or kCPU, kAUTO

    // Checkpointing
    cfg.checkpointSavePath = "checkpoints";
    cfg.timestepsPerCheckpoint = 1000000;
    cfg.checkpointLoadFolder = "";  // Empty for new training

    // Metrics
    cfg.sendMetrics = true;
    cfg.metricsSendRate = 5;      // Every 5 iterations
    cfg.renderSendRate = 0.5f;    // Every 0.5 seconds

    // Step 3: Create and start learner
    GGL::Learner* learner = new GGL::Learner(EnvCreateFunc, cfg);
    learner->Start();  // Blocks until training complete

    delete learner;
    return 0;
}
```

### 2. Creating Custom Rewards

**Pattern**: Inherit from `RLGC::Reward`

```cpp
// Example: Reward for staying close to ball
class BallProximityReward : public RLGC::Reward {
public:
    float GetReward(const RLGC::Player& player,
                   const RLGC::GameState& state,
                   bool isFinal) override {
        // Distance to ball
        float dist = (player.pos - state.ball.pos).Length();

        // Closer = better (inverse distance, clamped)
        float reward = 1.0f / (1.0f + dist / 1000.0f);

        // Bonus for getting closer over time
        if (player.prev) {
            float prevDist = (player.prev->pos - state.ball.pos).Length();
            if (dist < prevDist) {
                reward += 0.1f;  // Approaching bonus
            }
        }

        return reward;
    }
};
```

**Best Practices**:
- Keep rewards sparse (don't reward every tick)
- Use `isFinal` parameter for terminal rewards
- Access `player.prev` for temporal rewards
- Normalize reward scales across different reward components
- Use `CombinedReward` with weights for multi-objective

### 3. Creating Custom Observations

**Pattern**: Inherit from `RLGC::ObsBuilder`

```cpp
class MinimalObs : public RLGC::ObsBuilder {
private:
    static constexpr int OBS_SIZE = 35;

public:
    RLGC::FList BuildObs(const RLGC::Player& player,
                        const RLGC::GameState& state) override {
        RLGC::FList obs;
        obs.reserve(OBS_SIZE);

        // Ball state (6)
        obs.push_back(state.ball.pos.x / 4096.0f);
        obs.push_back(state.ball.pos.y / 6000.0f);
        obs.push_back(state.ball.pos.z / 2044.0f);
        obs.push_back(state.ball.vel.x / 6000.0f);
        obs.push_back(state.ball.vel.y / 6000.0f);
        obs.push_back(state.ball.vel.z / 6000.0f);

        // Player state (9)
        obs.push_back(player.pos.x / 4096.0f);
        obs.push_back(player.pos.y / 6000.0f);
        obs.push_back(player.pos.z / 2044.0f);
        obs.push_back(player.vel.x / 2300.0f);
        obs.push_back(player.vel.y / 2300.0f);
        obs.push_back(player.vel.z / 2300.0f);
        obs.push_back(player.boost / 100.0f);
        obs.push_back(player.isOnGround ? 1.0f : 0.0f);
        obs.push_back(player.hasFlip ? 1.0f : 0.0f);

        // ... add teammates, opponents, etc.

        return obs;
    }

    int GetObsSize() override { return OBS_SIZE; }
};
```

**Best Practices**:
- Normalize all values to reasonable ranges (typically [-1, 1] or [0, 1])
- Use game constants for normalization (max speed = 2300, field width = 8192, etc.)
- Include temporal information if needed (velocities, or use prev state)
- Consider symmetry (mirror observations for orange team)
- Pre-allocate with `reserve()` for performance

### 4. Implementing Action Masking

**Pattern**: Override `GetActionMask()` in `ActionParser`

```cpp
class MaskedActionParser : public RLGC::DiscreteAction {
public:
    std::vector<float> GetActionMask(const RLGC::Player& player,
                                     const RLGC::GameState& state) override {
        std::vector<float> mask(GetActionAmount(), 1.0f);

        // Mask jump if already jumping
        if (player.hasJumped && !player.hasFlip) {
            mask[ACTION_JUMP] = 0.0f;
            mask[ACTION_JUMP_FORWARD] = 0.0f;
            // ... mask all jump actions
        }

        // Mask boost if no boost available
        if (player.boost < 1.0f) {
            mask[ACTION_BOOST] = 0.0f;
            mask[ACTION_BOOST_FORWARD] = 0.0f;
            // ... mask all boost actions
        }

        return mask;
    }
};
```

**Important**: Action masking is properly implemented in GigaLearnCPP (unlike some other frameworks).

### 5. Loading and Deploying a Trained Model

**For RLBot Integration**:

```cpp
#include "RLBotClient.h"
#include <GigaLearnCPP/Util/InferUnit.h>

int main() {
    // Create observation builder and action parser
    // (must match training configuration)
    auto obsBuilder = new RLGC::DefaultObs();
    auto actionParser = new RLGC::DiscreteAction();

    // Load model configuration from checkpoint
    GGL::ModelConfig sharedHead, policy;
    sharedHead.layerSizes = {256, 256};
    policy.layerSizes = {256};

    // Create inference unit
    GGL::InferUnit* inferUnit = new GGL::InferUnit(
        obsBuilder,
        obsBuilder->GetObsSize(),
        actionParser,
        sharedHead,
        policy,
        "checkpoints/5000000/",  // Checkpoint folder
        true  // Use GPU (false for CPU)
    );

    // Setup RLBot
    RLBotParams params = {};
    params.port = 23233;  // From rlbot/port.cfg
    params.inferUnit = inferUnit;
    params.tickSkip = 8;  // Must match training

    // Run (blocking)
    RLBotClient::Run(params);

    delete inferUnit;
    return 0;
}
```

### 6. Monitoring Training with WandB

**Terminal 1** - Start metric receiver:
```bash
cd build/python_scripts
python metric_receiver.py
```

**Terminal 2** - Run training:
```bash
./GigaLearnBot
```

Metrics will automatically be sent to WandB (Weights & Biases):
- Project: `gigalearncpp`
- Run name: `gigalearncpp-run`

**Custom Metrics**:
```cpp
// In step callback
auto stepCallback = [](GGL::StepCallbackInfo& info) {
    // Add custom metrics
    info.report.AddAvg("MyMetric/Average", value);
    info.report.Set("MyMetric/Current", value);
};

cfg.stepCallback = stepCallback;
```

### 7. Checkpoint Management

**Auto-Saving**:
Checkpoints saved every `cfg.timestepsPerCheckpoint` steps to:
```
checkpoints/<timestep>/
├── PPO_POLICY.lt              # TorchScript policy model
├── PPO_POLICY_OPTIM.lt        # Policy optimizer state
├── PPO_CRITIC.lt              # TorchScript critic model
├── PPO_CRITIC_OPTIM.lt        # Critic optimizer state
├── PPO_SHAREDHEAD.lt          # Shared layers (if used)
├── PPO_SHAREDHEAD_OPTIM.lt    # Shared head optimizer
├── obs_stat.json              # Observation normalization stats
├── return_stat.json           # Return normalization stats
└── state.json                 # Timestep, iteration count
```

**Loading**:
```cpp
cfg.checkpointLoadFolder = "checkpoints/5000000/";
```

**Manual Save**:
```cpp
learner->SaveCheckpoint("manual_checkpoint/");
```

**Converting Checkpoints**:
```bash
python tools/checkpoint_converter.py \
    --input checkpoints/5000000/ \
    --output rlgym_ppo_checkpoint/
```

---

## Important Files Reference

### Must-Read Files

| File | Purpose | Priority |
|------|---------|----------|
| `/src/ExampleMain.cpp` | Complete training example, API demo | ⭐⭐⭐⭐⭐ |
| `/GigaLearnCPP/src/public/GigaLearnCPP/Learner.h` | Main API entry point | ⭐⭐⭐⭐⭐ |
| `/GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h` | Configuration reference | ⭐⭐⭐⭐⭐ |
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates/GameState.h` | State structure | ⭐⭐⭐⭐ |
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates/Player.h` | Player state | ⭐⭐⭐⭐ |
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Rewards/Reward.h` | Reward interface | ⭐⭐⭐⭐ |
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/OBSBuilders/ObsBuilder.h` | Obs interface | ⭐⭐⭐⭐ |

### Plugin Examples

| File | Purpose |
|------|---------|
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Rewards/CommonRewards.h` | Built-in reward implementations |
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/OBSBuilders/DefaultObs.h` | Default observation builder |
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/ActionParsers/DefaultAction.h` | Discrete action space |
| `/GigaLearnCPP/RLGymCPP/src/RLGymCPP/StateSetters/RandomState.h` | Random state initialization |

### Utilities

| File | Purpose |
|------|---------|
| `/GigaLearnCPP/src/public/GigaLearnCPP/Util/InferUnit.h` | Deployment inference |
| `/GigaLearnCPP/src/public/GigaLearnCPP/Util/Report.h` | Metrics reporting |
| `/GigaLearnCPP/src/public/GigaLearnCPP/Util/Timer.h` | Performance timing |
| `/src/RLBotClient.h` | RLBot integration |

---

## Advanced Features

### 1. Shared Layers

Shared layers between policy and critic networks improve sample efficiency:

```cpp
cfg.ppo.sharedHead.layerSizes = {256, 256};  // Shared backbone
cfg.ppo.policy.layerSizes = {128};           // Policy-specific head
cfg.ppo.critic.layerSizes = {128};           // Critic-specific head
```

**Recommendation**: Enable by default (as shown above).

### 2. ELO Skill Tracking

Track agent skill over time using ELO ratings:

```cpp
cfg.skillTracker.enabled = true;
cfg.skillTracker.k = 32.0f;           // ELO K-factor
cfg.skillTracker.defaultMMR = 1200.0f;
cfg.policyVersionsPerCheckpoint = 5;  // Save versioned policies
```

Allows training against older versions (coming soon).

### 3. Layer Normalization

Stabilizes training (highly recommended):

```cpp
cfg.ppo.policy.layerNorm = true;
cfg.ppo.critic.layerNorm = true;
cfg.ppo.sharedHead.layerNorm = true;
```

### 4. Custom Optimizers

```cpp
cfg.ppo.policy.optimizer = GGL::EOptimizer::ADAM;
cfg.ppo.critic.optimizer = GGL::EOptimizer::ADAMW;

// Available: ADAM, ADAMW, ADAGRAD, RMSPROP, MAGSGD
```

### 5. Custom Activation Functions

```cpp
cfg.ppo.policy.activation = GGL::EActivation::RELU;

// Available: RELU, LEAKYRELU, SIGMOID, TANH
```

### 6. Action Delay

Simulate input lag (realistic for online play):

```cpp
cfg.actionDelay = 1;  // 1 tick delay (~16ms at 60 TPS)
```

### 7. Half-Precision Inference

Faster inference with minimal accuracy loss:

```cpp
cfg.ppo.halfPrecisionInference = true;
```

### 8. Transfer Learning

Load policy from previous run, initialize new critic:

```cpp
GGL::TransferLearnConfig transfer = {};
transfer.loadPolicy = true;
transfer.loadCritic = false;
transfer.loadSharedHead = true;
transfer.checkpointPath = "old_checkpoint/";

cfg.ppo.transferLearn = transfer;
```

---

## Troubleshooting

### Common Issues

**Issue**: "LibTorch not found"
- **Solution**: Set `CMAKE_PREFIX_PATH` to LibTorch directory
- Verify `libtorch/share/cmake/Torch/TorchConfig.cmake` exists

**Issue**: "Cannot find collision mesh"
- **Solution**: Dump `.cmf` files from Rocket League
- Use `Arena::Create(mode, "path/to/soccar.cmf")`

**Issue**: Training crashes with CUDA errors
- **Solution**:
  - Check CUDA version matches LibTorch build
  - Reduce batch size if out of memory
  - Try `cfg.device = torch::kCPU` to isolate GPU issues

**Issue**: "Python module not found" (metrics)
- **Solution**: Install WandB: `pip install wandb`
- Or disable metrics: `cfg.sendMetrics = false`

**Issue**: Slow training performance
- **Solution**:
  - Increase `cfg.numGames` (more parallel envs)
  - Enable `cfg.ppo.halfPrecisionInference = true`
  - Reduce PPO epochs or batch size
  - Use GPU: `cfg.device = torch::kCUDA`

**Issue**: Agent not learning / poor performance
- **Solution**:
  - Check reward function (should be non-zero)
  - Verify observation normalization ranges
  - Try simpler model first (fewer layers)
  - Increase entropy coefficient (more exploration)
  - Check terminal conditions (episodes ending too soon?)

### Debugging Tips

**Enable verbose logging**:
```cpp
RG_LOG("Debug message: " << value);
```

**Monitor metrics**:
- Watch `SPS` (steps per second) - should be high (>10k)
- Watch policy loss, value loss convergence
- Watch entropy (should decrease slowly over time)

**Visualize training**:
```cpp
cfg.renderSendRate = 1.0f;  // Send render every 1 second
```
Run `python render_receiver.py` to visualize.

**Profile performance**:
```cpp
GGL::Timer timer;
timer.Reset();
// ... code to profile
float elapsed = timer.Elapsed();
RG_LOG("Took " << elapsed << " seconds");
```

---

## Development Guidelines for AI Assistants

### When Reading Code

1. **Start with ExampleMain.cpp** - Comprehensive usage example
2. **Check LearnerConfig.h** - Understand all configuration options
3. **Read plugin interfaces** - Reward.h, ObsBuilder.h, etc.
4. **Understand state structure** - GameState.h, Player.h

### When Modifying Code

1. **Prefer editing existing files** over creating new ones
2. **Follow namespace conventions** (GGL, RLGC, RS)
3. **Match existing code style** (PascalCase methods, camelCase vars)
4. **Don't modify framework internals** unless absolutely necessary
5. **Test changes with small configs** before full training runs

### When Helping Users

1. **Ask about their goal** - Training? Deployment? Custom reward?
2. **Check their setup** - LibTorch installed? Collision meshes?
3. **Start simple** - Get basic example working first
4. **Reference ExampleMain.cpp** - Show concrete examples
5. **Explain tradeoffs** - Performance vs. sample efficiency, etc.

### Common User Requests

**"How do I train an agent?"**
- Point to ExampleMain.cpp
- Walk through EnvCreateFunc, config, Learner creation
- Explain build process

**"How do I create a custom reward?"**
- Show Reward interface
- Provide simple example (proximity, velocity, etc.)
- Explain `player.prev` for temporal rewards
- Recommend using CombinedReward with weights

**"My agent won't learn"**
- Check reward function (is it zero?)
- Verify observations are normalized
- Suggest simpler architecture initially
- Check terminal conditions

**"How do I deploy my trained model?"**
- Explain InferUnit usage
- Show RLBotClient example
- Emphasize matching obs/action parsers

**"Training is slow"**
- Increase `numGames` (parallel envs)
- Enable half-precision inference
- Check if GPU is being used
- Profile with Timer

### Key Concepts to Emphasize

1. **Previous State Access** - Unique feature, enables temporal rewards
2. **Action Masking** - Properly implemented, explain how to use
3. **Plugin Architecture** - User implements interfaces, framework calls them
4. **Checkpoint Structure** - Explain folder layout, what each file is
5. **Performance** - This is a fast framework, leverage parallelism

---

## Quick Reference

### Typical Training Configuration

```cpp
GGL::LearnerConfig cfg = {};
cfg.numGames = 256;
cfg.timestepsPerIteration = 50000;
cfg.tickSkip = 8;
cfg.ppo.batchSize = 50000;
cfg.ppo.epochs = 30;
cfg.ppo.policyLR = 5e-5f;
cfg.ppo.criticLR = 5e-5f;
cfg.ppo.sharedHead.layerSizes = {256, 256};
cfg.ppo.policy.layerSizes = {256};
cfg.ppo.critic.layerSizes = {256};
cfg.ppo.policy.layerNorm = true;
cfg.device = torch::kCUDA;
```

### Common Rocket League Constants

```cpp
// Field dimensions (units)
constexpr float FIELD_HALF_LENGTH = 5120.0f;
constexpr float FIELD_HALF_WIDTH = 4096.0f;
constexpr float FIELD_HEIGHT = 2044.0f;
constexpr float GOAL_HEIGHT = 642.775f;

// Car physics
constexpr float MAX_CAR_SPEED = 2300.0f;  // uu/s
constexpr float BOOST_MAX = 100.0f;
constexpr float BOOST_CONSUMPTION = 33.3f;  // per second at full throttle

// Ball physics
constexpr float BALL_RADIUS = 91.25f;
constexpr float BALL_MAX_SPEED = 6000.0f;

// Boost pads
constexpr int NUM_BOOST_PADS = 34;
constexpr int NUM_BIG_PADS = 6;
constexpr int NUM_SMALL_PADS = 28;
```

### File Locations Cheat Sheet

```
Core API:         /GigaLearnCPP/src/public/GigaLearnCPP/
Implementation:   /GigaLearnCPP/src/private/GigaLearnCPP/
Environment:      /GigaLearnCPP/RLGymCPP/src/RLGymCPP/
Physics:          /GigaLearnCPP/RLGymCPP/RocketSim/
RLBot:            /RLBotCPP/
Examples:         /src/
Python scripts:   /GigaLearnCPP/python_scripts/
Tools:            /tools/
```

---

## Additional Resources

### External Documentation

- **PyTorch C++ API**: https://pytorch.org/cppdocs/
- **RLBot Framework**: https://github.com/RLBot/RLBot
- **RocketSim**: https://github.com/ZealanL/RocketSim
- **PPO Paper**: https://arxiv.org/abs/1707.06347

### Community

- RLBot Discord: https://discord.gg/rlbot
- Rocket League AI community resources

---

## Version Information

**Last Updated**: Based on leaked version (commit: 86e11c3)
**Framework Version**: GigaLearnCPP (no official version number)
**C++ Standard**: C++20
**Build System**: CMake 3.8+

---

This guide should provide AI assistants with comprehensive context for understanding and working with the GigaLearnCPP codebase. For specific implementation questions, always refer to `/src/ExampleMain.cpp` as the canonical usage example.
