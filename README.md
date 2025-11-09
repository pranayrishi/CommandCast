# CommandCast: AI-Powered Mac Control via Voice

Control your entire Mac with AI voice agents through native Apple interfaces:

1. **FaceTime**: Call your Mac to start a FaceTime session with screen sharing. Talk naturally to execute any computer task.
2. **iMessage**: Text commands to your Mac and have them fulfilled automatically.

Built on the Agent-S framework, this project enables seamless remote control and automation of macOS through voice commands and natural language processing.

## Features

- **Voice-Controlled Mac Automation**: Execute complex tasks on your Mac using natural language
- **FaceTime Integration**: Remote control via FaceTime with screen sharing
- **iMessage Commands**: Send text commands for automated task execution
- **Agent-S Framework**: Leverages state-of-the-art computer-use AI agent technology
- **Native Apple Integration**: No third-party remote desktop apps required

## Project Structure

The project is organized into three main components:

1. **`Agent-S/`** — Fork of the state-of-the-art computer-use agent framework ([Original Repo](https://github.com/simular-ai/Agent-S))
2. **`backend/`** — Flask server handling iMessage/FaceTime integration, voice transcription, and AI responses
3. **`frontend/`** — Electron-based UI for monitoring and controlling Agent-S actions


## Quick Start

All you need is a single LLM key. Export `OPENAI_API_KEY` (or swap in the key for your preferred provider) and you’re ready.

**1. Install dependencies**

```bash
git clone https://github.com/pranayrishi/CommandCast.git
cd CommandCast

# Setup Agent-S (see original repo for more details/debugging)
cd Agent-S
uv sync

cd ..

# Setup backend
cd backend
uv sync

cd ..

# Setup UI
cd frontend
SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm install --ignore-scripts
npm rebuild sharp
```

For more details on Agent S: https://github.com/simular-ai/Agent-S

**2. Configure API Keys**

Create a `.env` file in the `backend/` directory with your API keys:

```bash
# Required: LLM Provider (choose one)
GROK_API_KEY=xai-your-grok-key
# OR
OPENAI_API_KEY=sk-your-openai-key

# Optional: For Text-to-Speech and Speech-to-Text
FISH_API_KEY=your-fish-api-key
```

Get your Fish Audio API key at: https://fish.audio/app/api-keys/

**Note**: Never commit `.env` files to version control. They are already excluded via `.gitignore`.

**3. Give Agent Permission to Control Keyboard/Mouse**

When you launch for the first time (see final step), you will be prompted to give permissions to `Terminal` or `VS Code`, etc. This is required for the Agent to control your computer.

Grant the following permissions in System Preferences:
- **Accessibility**: Allow Terminal/VS Code to control your computer
- **Automation**: Allow the app to control other applications
- **Disk Access**: Grant full disk access for file operations

**4. Route FaceTime audio input/output (optional)**

To route audio directly from FaceTime to the AI Agent, install BlackHole audio driver. This is optional if you want to use iMessage only.

Steps:
1. Install BlackHole App (both 2ch and 16ch versions): https://github.com/ExistentialAudio/BlackHole
2. Restart your computer
3. In FaceTime menu, set `Video → Microphone` to **BlackHole 2ch**
4. Set `Video → Output` to **BlackHole 16ch**

**5. Launch CommandCast**

In the base directory:

```bash
# Run everything (UI, backend, Agent-S)
./run.sh
```

## Why CommandCast?

### Seamless Remote Control

Control your Mac through native Apple interfaces — FaceTime and iMessage. No clunky remote desktop apps, no third-party tools, just the simplicity of built-in macOS features.

### Intelligent Automation

Powered by the Agent-S framework, CommandCast enables sophisticated computer-use tasks through natural language. Execute complex workflows, automate repetitive tasks, and control your Mac from anywhere.

### Privacy & Security

- All API keys stored in `.env` files (never committed to version control)
- Local processing where possible
- Full control over what permissions the agent has

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project builds upon the Agent-S framework. Please refer to individual component licenses for details.

## Acknowledgments

- Built on top of [Agent-S](https://github.com/simular-ai/Agent-S) framework
- Inspired by the need for seamless Mac automation and remote control
