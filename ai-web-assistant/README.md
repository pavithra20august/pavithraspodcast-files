# AI Web Assistant

A powerful browser extension that acts as an AI-powered assistant to understand website content and perform tasks based on natural language instructions. Similar to Comet browser functionality, but as a standalone extension powered by Google's Gemini AI.

## Features

🤖 **AI-Powered Understanding**: Uses Gemini 2.0 Flash to understand website content and user intentions

🎯 **Task Automation**: Automatically performs web tasks like:
- Filling out forms
- Clicking buttons and links
- Navigating websites
- Extracting and summarizing content

🔍 **Smart Content Analysis**: Intelligently extracts and analyzes:
- Page content and structure
- Available forms and fields
- Interactive elements
- Metadata and structured data

💬 **Natural Language Interface**: Simply describe what you want to do in plain English

## Installation

1. **Download the Extension**
   - Download or clone this repository to your computer

2. **Load in Chrome/Edge**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" in the top right
   - Click "Load unpacked" and select the `ai-web-assistant` folder

3. **Setup API Key** (Already configured)
   - The extension comes pre-configured with a Gemini API key
   - Uses `gemini-2.0-flash-exp` model for optimal performance

## Usage

### Basic Usage

1. **Click the Extension Icon** in your browser toolbar
2. **Enter Your Instruction** in natural language
3. **Click "Execute Task"** or press Ctrl+Enter

### Example Instructions

📝 **Form Filling**
```
Fill out the contact form with: Name: John Doe, Email: john@example.com
```

🔍 **Finding and Clicking**
```
Find and click the 'Sign Up' button
Find the cheapest product and add it to cart
```

📊 **Content Analysis**
```
Summarize the main content of this page in 3 bullet points
Find all the pricing information on this page
```

🛒 **E-commerce Tasks**
```
Add the first item to my shopping cart
Find products under $50 and show me the options
```

### Page Analysis

Click the **"Analyze Page"** button to:
- Get an overview of page content
- See available forms and interactive elements
- Highlight clickable elements on the page

## How It Works

1. **Content Extraction**: The extension extracts all relevant content from the webpage including text, forms, buttons, and links

2. **AI Processing**: Your instruction and the page content are sent to Gemini AI for analysis

3. **Action Planning**: Gemini determines the best sequence of actions to complete your task

4. **Task Execution**: The extension automatically performs the planned actions on the webpage

## Capabilities

### ✅ What It Can Do

- Fill out any form fields with provided information
- Click buttons, links, and other interactive elements
- Navigate through multi-step processes
- Extract and summarize page content
- Find specific information on pages
- Interact with e-commerce sites
- Handle dynamic content and modern web apps

### ❌ Limitations

- Cannot handle CAPTCHAs or anti-bot measures
- Cannot perform actions requiring human verification
- Limited to single-page tasks (doesn't navigate across domains)
- Cannot access content behind login walls (unless already logged in)
- Respects website security policies and CORS restrictions

## Privacy & Security

- 🔒 **No Data Storage**: The extension doesn't store any personal information
- 🌐 **API Only**: Communication only with Google's Gemini API
- 🛡️ **Safe Actions**: Conservative approach to potentially destructive actions
- 👀 **Transparent**: All actions are logged and visible to the user

## Troubleshooting

### Extension Not Working
1. Refresh the webpage and try again
2. Check that you're on a regular website (not chrome:// pages)
3. Ensure the extension has permission to access the site

### AI Not Understanding Instructions
1. Be more specific in your instructions
2. Use the "Analyze Page" feature first to understand available elements
3. Break complex tasks into smaller steps

### API Errors
- The extension uses a pre-configured API key for Gemini 2.5 Pro
- If you encounter rate limits, wait a few minutes and try again
- Check browser console (F12) for detailed error messages
- Use the included `test-api.html` file to verify API connectivity

### Common API Error Solutions
- **403 Forbidden**: API key invalid or expired
- **404 Not Found**: Model name incorrect (should be `gemini-2.5-pro`)
- **429 Too Many Requests**: Rate limit exceeded, wait and retry
- **400 Bad Request**: Invalid request format or content

## Development

### Project Structure
```
ai-web-assistant/
├── manifest.json          # Extension configuration
├── popup.html             # Extension popup interface
├── popup.js               # Popup logic and UI interactions
├── content.js             # Content script for page interaction
├── background.js          # Background service with Gemini AI integration
└── README.md              # This file
```

### Key Components

- **Content Script**: Extracts page content and executes actions
- **Background Script**: Handles Gemini API communication
- **Popup Interface**: User interaction and instruction input

## API Configuration

The extension is pre-configured with:
- **API Key**: `gemini api key`
- **Model**: `model name`
- **Endpoint**: Google's Generative Language API

### Testing API Configuration

To verify your API configuration is working:
1. Open `test-api.html` in your browser
2. Click "Test API Connection"
3. Check for successful connection and response

## Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

This project is open source. Use responsibly and respect website terms of service.

---

**Powered by Google Gemini AI** 🚀