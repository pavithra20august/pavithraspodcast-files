// Popup script for the AI Web Assistant
class PopupManager {
  constructor() {
    this.currentTabId = null;
    this.currentTab = null;
    this.isExecuting = false;
    this.init();
  }

  async init() {
    await this.getCurrentTab();
    this.checkTabCompatibility();
    this.setupEventListeners();
    this.setupExamples();
  }

  async getCurrentTab() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      this.currentTab = tab;
      this.currentTabId = tab.id;
      console.log('Current tab:', tab.url);
    } catch (error) {
      console.error('Error getting current tab:', error);
      this.showStatus('❌ Could not access current tab', 'error');
    }
  }

  checkTabCompatibility() {
    if (!this.currentTab) {
      this.showStatus('❌ Cannot access current tab', 'error');
      return false;
    }

    const url = this.currentTab.url;

    // Check for incompatible URLs
    if (url.startsWith('chrome://') ||
        url.startsWith('chrome-extension://') ||
        url.startsWith('edge://') ||
        url.startsWith('about:') ||
        url.startsWith('moz-extension://') ||
        url === 'chrome://newtab/' ||
        url === 'edge://newtab/') {

      this.showStatus('❌ Cannot work on browser internal pages. Please navigate to a regular website.', 'error');
      document.getElementById('executeBtn').disabled = true;
      document.getElementById('analyzeBtn').disabled = true;
      return false;
    }

    return true;
  }

  setupEventListeners() {
    const executeBtn = document.getElementById('executeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const instructionInput = document.getElementById('instruction');

    executeBtn.addEventListener('click', () => this.executeTask());
    analyzeBtn.addEventListener('click', () => this.analyzePage());

    // Enter key to execute
    instructionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.ctrlKey) {
        this.executeTask();
      }
    });

    // Auto-focus on instruction input
    instructionInput.focus();
  }

  setupExamples() {
    const examples = document.querySelectorAll('.example');
    const instructionInput = document.getElementById('instruction');

    examples.forEach(example => {
      example.addEventListener('click', () => {
        const text = example.getAttribute('data-text');
        instructionInput.value = text;
        instructionInput.focus();
      });
    });
  }

  async ensureContentScript() {
    try {
      // Try to ping the content script
      const response = await chrome.tabs.sendMessage(this.currentTabId, { action: 'ping' });
      console.log('Content script is ready:', response);
      return true;
    } catch (error) {
      console.log('Content script not found, injecting...');

      try {
        // Inject the content script
        await chrome.scripting.executeScript({
          target: { tabId: this.currentTabId },
          files: ['content.js']
        });

        // Wait a bit for the script to initialize
        await this.sleep(500);

        // Try to ping again
        const response = await chrome.tabs.sendMessage(this.currentTabId, { action: 'ping' });
        console.log('Content script injected and ready:', response);
        return true;
      } catch (injectError) {
        console.error('Failed to inject content script:', injectError);
        throw new Error('Cannot inject content script. Make sure you have permission to access this page.');
      }
    }
  }

  async executeTask() {
    if (this.isExecuting) return;

    const instruction = document.getElementById('instruction').value.trim();
    if (!instruction) {
      this.showStatus('Please enter an instruction', 'error');
      return;
    }

    if (!this.checkTabCompatibility()) {
      return;
    }

    this.isExecuting = true;
    this.setButtonsState(false);
    this.showStatus('🤖 Processing your request...', 'info', true);

    try {
      // Ensure content script is loaded
      await this.ensureContentScript();

      // Send message to content script to execute the task
      const response = await chrome.tabs.sendMessage(this.currentTabId, {
        action: 'executeTask',
        instruction: instruction
      });

      if (response && response.success) {
        this.showStatus(`✅ ${response.message}`, 'success');

        // Clear instruction after successful execution
        setTimeout(() => {
          document.getElementById('instruction').value = '';
        }, 2000);
      } else {
        const errorMsg = response?.error || 'Failed to execute task';
        this.showStatus(`❌ ${errorMsg}`, 'error');
      }
    } catch (error) {
      console.error('Error executing task:', error);

      if (error.message.includes('Could not establish connection')) {
        this.showStatus('❌ Cannot communicate with page. Try refreshing the page and reopening the extension.', 'error');
      } else if (error.message.includes('permission')) {
        this.showStatus('❌ No permission to access this page. The extension may be blocked on this site.', 'error');
      } else {
        this.showStatus(`❌ ${error.message}`, 'error');
      }
    } finally {
      this.isExecuting = false;
      this.setButtonsState(true);
    }
  }

  async analyzePage() {
    if (!this.checkTabCompatibility()) {
      return;
    }

    this.setButtonsState(false);
    this.showStatus('🔍 Analyzing page content...', 'info', true);

    try {
      // Ensure content script is loaded
      await this.ensureContentScript();

      // Send message to content script to extract content
      const response = await chrome.tabs.sendMessage(this.currentTabId, {
        action: 'extractContent'
      });

      if (response && response.success) {
        const content = response.content;
        let analysis = `📊 Page Analysis:\n\n`;
        analysis += `📄 Title: ${content.title}\n`;
        analysis += `🔗 URL: ${content.url}\n`;
        analysis += `📝 Text length: ${content.text.length} characters\n`;
        analysis += `📋 Forms: ${content.forms.length}\n`;
        analysis += `🔗 Links: ${content.links.length}\n`;
        analysis += `🔲 Buttons: ${content.buttons.length}\n`;

        // Add loading status information
        if (content.loadingStatus) {
          analysis += `\n🔄 Page Status:\n`;
          analysis += `  • Ready State: ${content.loadingStatus.readyState}\n`;
          analysis += `  • Is Loading: ${content.loadingStatus.isLoading ? 'Yes' : 'No'}\n`;
          analysis += `  • Has Errors: ${content.loadingStatus.hasError ? 'Yes' : 'No'}\n`;
          if (content.loadingStatus.hasError) {
            analysis += `  ⚠️ Page may not be fully loaded. Try refreshing or waiting longer.\n`;
          }
        }

        if (content.forms.length > 0) {
          analysis += `\n🏷️ Available form fields:\n`;
          content.forms[0].fields.slice(0, 5).forEach(field => {
            analysis += `  • ${field.label || field.name || field.type}\n`;
          });
        }

        if (content.buttons.length > 0) {
          analysis += `\n🔘 Key buttons:\n`;
          content.buttons.slice(0, 5).forEach(button => {
            if (button.text.trim()) {
              analysis += `  • ${button.text.trim()}\n`;
            }
          });
        }

        this.showStatus(analysis, 'info');

        // Highlight interactable elements on the page
        try {
          await chrome.tabs.sendMessage(this.currentTabId, {
            action: 'highlightElements'
          });

          setTimeout(async () => {
            try {
              await chrome.tabs.sendMessage(this.currentTabId, {
                action: 'clearHighlight'
              });
            } catch (e) {
              // Ignore errors when clearing highlights
            }
          }, 5000);
        } catch (e) {
          // Highlighting failed, but analysis succeeded
          console.log('Could not highlight elements:', e);
        }

      } else {
        this.showStatus('❌ Failed to analyze page', 'error');
      }
    } catch (error) {
      console.error('Error analyzing page:', error);

      if (error.message.includes('Could not establish connection')) {
        this.showStatus('❌ Cannot communicate with page. Try refreshing the page and reopening the extension.', 'error');
      } else {
        this.showStatus(`❌ Error analyzing page: ${error.message}`, 'error');
      }
    } finally {
      this.setButtonsState(true);
    }
  }

  setButtonsState(enabled) {
    const executeBtn = document.getElementById('executeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');

    executeBtn.disabled = !enabled;
    analyzeBtn.disabled = !enabled;

    if (!enabled) {
      executeBtn.style.opacity = '0.6';
      analyzeBtn.style.opacity = '0.6';
    } else {
      executeBtn.style.opacity = '1';
      analyzeBtn.style.opacity = '1';
    }
  }

  showStatus(message, type = 'info', showLoading = false) {
    const statusDiv = document.getElementById('status');
    statusDiv.className = `status ${type}`;

    if (showLoading) {
      statusDiv.innerHTML = `<div class="loading"></div>${message}`;
    } else {
      statusDiv.innerHTML = message.replace(/\n/g, '<br>');
    }

    statusDiv.style.display = 'block';

    // Don't auto-hide messages - let user read them
    // Users can manually close by clicking buttons or opening new actions
  }

  hideStatus() {
    document.getElementById('status').style.display = 'none';
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new PopupManager();
});