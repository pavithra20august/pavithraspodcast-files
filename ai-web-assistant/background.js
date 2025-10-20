// Background script with Gemini API integration
class AIWebAssistantBackground {
  constructor() {
    this.apiKey = 'gemini api key';
    this.model = 'model name';
    this.apiUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent';
    this.setupMessageListener();
  }

  setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === 'processWithAI') {
        this.processInstructionWithAI(request.instruction, request.content)
          .then(result => sendResponse(result))
          .catch(error => {
            console.error('AI processing error:', error);
            sendResponse({
              success: false,
              error: error.message || 'Failed to process instruction'
            });
          });
        return true; // Keep message channel open for async response
      }
    });
  }

  async processInstructionWithAI(instruction, pageContent) {
    try {
      const prompt = this.buildPrompt(instruction, pageContent);
      const response = await this.callGeminiAPI(prompt);

      if (response && response.candidates && response.candidates[0]) {
        const content = response.candidates[0].content.parts[0].text;
        return this.parseAIResponse(content);
      } else {
        throw new Error('Invalid response from Gemini API');
      }
    } catch (error) {
      console.error('Error processing with AI:', error);
      throw error;
    }
  }

  buildPrompt(instruction, pageContent) {
    return `You are an AI web assistant that helps users interact with websites. You can analyze page content and generate actions to complete user tasks.

WEBSITE CONTENT:
Title: ${pageContent.title}
URL: ${pageContent.url}
Main Text: ${pageContent.text.substring(0, 3000)}...

${pageContent.loadingStatus?.hasError ?
  'WARNING: Page appears to have loading issues or errors. Content may be incomplete.' : ''}
${pageContent.loadingStatus?.isLoading ?
  'WARNING: Page appears to still be loading. Content may be incomplete.' : ''}

Available Forms: ${JSON.stringify(pageContent.forms, null, 2)}
Available Buttons: ${JSON.stringify(pageContent.buttons, null, 2)}
Available Links: ${JSON.stringify(pageContent.links.slice(0, 10), null, 2)}

USER INSTRUCTION: "${instruction}"

Based on the user's instruction and the website content, you need to:

1. Analyze what the user wants to achieve
2. Determine if the task is possible with the available page elements
3. Generate a sequence of actions to complete the task

RESPONSE FORMAT (JSON):
{
  "analysis": "Brief explanation of what you understood and plan to do",
  "feasible": true/false,
  "actions": [
    {
      "type": "click|fill|submit|scroll|highlight",
      "selector": "CSS selector for the element",
      "value": "value to fill (for fill type only)",
      "description": "Human-readable description of the action"
    }
  ],
  "message": "User-friendly message about what was completed or why it failed"
}

ACTION TYPES:
- "click": Click on buttons, links, or interactive elements
- "fill": Fill form fields with values
- "submit": Submit forms
- "scroll": Scroll to bring elements into view
- "highlight": Highlight elements for user attention

GUIDELINES:
- Only use selectors from the provided page elements
- The system now supports multiple interaction methods (click, fill, scroll, highlight)
- For form filling, it can handle: regular inputs, contenteditable elements, rich text editors, and Google Docs
- Text input methods include: standard form fields, rich text editors, and Google Docs-specific editing
- Button clicking uses multiple fallback methods for better compatibility
- Be conservative - if you're not sure about an action, explain why in the message
- If the task isn't possible, set feasible to false and explain why
- Always provide clear, helpful messages to the user
- For complex tasks, break them down into simple sequential actions
- Prioritize user safety - don't perform destructive actions without clear intent
- For rich text editors and Google Docs, use "fill" action type with the appropriate selector
- The system can find elements even if selectors don't match exactly (fuzzy matching)

SUPPORTED ELEMENT TYPES:
- Regular form inputs (input, textarea, select)
- Contenteditable elements (rich text editors)
- Google Docs/Sheets editors
- Buttons (including div/span elements with click handlers)
- Links used as buttons
- Elements with ARIA roles

Example for "Fill contact form with name John Doe, email john@example.com":
{
  "analysis": "User wants to fill a contact form with provided details",
  "feasible": true,
  "actions": [
    {
      "type": "fill",
      "selector": "input[name='name']",
      "value": "John Doe",
      "description": "Fill name field with 'John Doe'"
    },
    {
      "type": "fill",
      "selector": "input[name='email']",
      "value": "john@example.com",
      "description": "Fill email field with 'john@example.com'"
    }
  ],
  "message": "Filled contact form with the provided name and email"
}

Return ONLY the JSON response, no additional text.`;
  }

  async callGeminiAPI(prompt) {
    const requestBody = {
      contents: [{
        parts: [{
          text: prompt
        }]
      }],
      generationConfig: {
        temperature: 0.1,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 2048,
        stopSequences: []
      },
      safetySettings: [
        {
          category: "HARM_CATEGORY_HARASSMENT",
          threshold: "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
          category: "HARM_CATEGORY_HATE_SPEECH",
          threshold: "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
          category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          threshold: "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
          category: "HARM_CATEGORY_DANGEROUS_CONTENT",
          threshold: "BLOCK_MEDIUM_AND_ABOVE"
        }
      ]
    };

    console.log('Making API call to:', `${this.apiUrl}?key=${this.apiKey.substring(0, 10)}...`);

    try {
      const response = await fetch(`${this.apiUrl}?key=${this.apiKey}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      console.log('API Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        let errorData;
        try {
          errorData = JSON.parse(errorText);
        } catch {
          errorData = { error: { message: errorText } };
        }

        console.error('API Error Details:', errorData);

        if (response.status === 400) {
          throw new Error(`Invalid request: ${errorData.error?.message || 'Bad request format'}`);
        } else if (response.status === 403) {
          throw new Error(`API key error: ${errorData.error?.message || 'Invalid or expired API key'}`);
        } else if (response.status === 404) {
          throw new Error(`Model not found: ${errorData.error?.message || 'The specified model does not exist'}`);
        } else if (response.status === 429) {
          throw new Error(`Rate limit exceeded: ${errorData.error?.message || 'Too many requests'}`);
        } else {
          throw new Error(`Gemini API error (${response.status}): ${errorData.error?.message || 'Unknown error'}`);
        }
      }

      const responseData = await response.json();
      console.log('API Response received successfully');
      return responseData;

    } catch (error) {
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Network error: Unable to connect to Gemini API. Check your internet connection.');
      }
      throw error;
    }
  }

  parseAIResponse(content) {
    try {
      console.log('Parsing AI response:', content.substring(0, 200) + '...');

      // Clean the content - remove markdown code blocks if present
      let cleanContent = content.trim();
      if (cleanContent.startsWith('```json')) {
        cleanContent = cleanContent.slice(7);
      }
      if (cleanContent.startsWith('```')) {
        cleanContent = cleanContent.slice(3);
      }
      if (cleanContent.endsWith('```')) {
        cleanContent = cleanContent.slice(0, -3);
      }
      cleanContent = cleanContent.trim();

      // Try to extract JSON from the content if it contains other text
      const jsonMatch = cleanContent.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        cleanContent = jsonMatch[0];
      }

      const parsed = JSON.parse(cleanContent);

      // Validate the response structure
      if (!parsed.hasOwnProperty('feasible')) {
        throw new Error('Response missing "feasible" property');
      }

      if (!parsed.hasOwnProperty('message')) {
        throw new Error('Response missing "message" property');
      }

      if (parsed.feasible && (!parsed.actions || !Array.isArray(parsed.actions))) {
        console.warn('Feasible task but missing actions array, adding empty array');
        parsed.actions = [];
      }

      console.log('Successfully parsed AI response');

      return {
        success: true,
        analysis: parsed.analysis || 'No analysis provided',
        feasible: parsed.feasible,
        actions: parsed.actions || [],
        message: parsed.message
      };

    } catch (error) {
      console.error('Error parsing AI response:', error);
      console.error('Raw content:', content);

      // Try to extract a meaningful error message
      let errorMessage = 'Failed to parse AI response.';

      if (content.includes('blocked') || content.includes('safety')) {
        errorMessage = 'Request was blocked by safety filters. Try rephrasing your instruction.';
      } else if (content.includes('error') || content.includes('Error')) {
        errorMessage = 'AI encountered an error processing your request.';
      } else if (!content.trim()) {
        errorMessage = 'AI returned an empty response. Please try again.';
      }

      // Fallback response
      return {
        success: false,
        error: errorMessage
      };
    }
  }

  // Utility method to validate selectors
  isValidSelector(selector) {
    try {
      document.createElement('div').querySelector(selector);
      return true;
    } catch {
      return false;
    }
  }
}

// Initialize the background service
const aiAssistant = new AIWebAssistantBackground();

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('AI Web Assistant installed successfully!');

    // Set default settings
    chrome.storage.sync.set({
      apiKey: 'gemini api key',
      model: 'model name',
      enableLogging: false
    });
  }
});