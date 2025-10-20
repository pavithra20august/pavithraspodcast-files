// Content script that runs on all web pages
class WebAssistant {
  constructor() {
    this.isHighlighting = false;
    this.highlightedElements = new Set();
    this.setupMessageListener();
  }

  setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      switch (request.action) {
        case 'ping':
          sendResponse({ success: true, message: 'Content script is ready' });
          break;

        case 'extractContent':
          this.extractPageContent().then(content => {
            sendResponse({ success: true, content });
          });
          return true; // Keep message channel open for async response

        case 'executeTask':
          this.executeTask(request.instruction).then(result => {
            sendResponse(result);
          });
          return true;

        case 'highlightElements':
          this.highlightInteractableElements();
          sendResponse({ success: true });
          break;

        case 'clearHighlight':
          this.clearHighlights();
          sendResponse({ success: true });
          break;

        case 'clickElement':
          this.clickElementBySelector(request.selector);
          sendResponse({ success: true });
          break;

        case 'fillForm':
          this.fillFormFields(request.fields);
          sendResponse({ success: true });
          break;
      }
    });
  }

  async extractPageContent() {
    // Wait for dynamic content to load
    await this.waitForContent();

    const content = {
      url: window.location.href,
      title: document.title,
      text: this.getMainTextContent(),
      forms: this.extractForms(),
      links: this.extractLinks(),
      buttons: this.extractButtons(),
      images: this.extractImages(),
      metadata: this.extractMetadata(),
      loadingStatus: this.checkPageLoadingStatus()
    };

    return content;
  }

  async waitForContent() {
    // Wait for basic DOM content
    if (document.readyState !== 'complete') {
      await new Promise(resolve => {
        if (document.readyState === 'complete') {
          resolve();
        } else {
          window.addEventListener('load', resolve, { once: true });
        }
      });
    }

    // Additional wait for dynamic content (especially for SPAs like Google Chat)
    await this.sleep(2000);

    // Wait for specific indicators that content is loaded
    const indicators = [
      '[data-thread-id]', // Google Chat messages
      '.chat-message',
      '.message',
      '[role="main"]',
      '.content-loaded',
      '.messages-container'
    ];

    for (let i = 0; i < 5; i++) { // Try for up to 5 seconds
      const hasContent = indicators.some(selector => document.querySelector(selector));
      if (hasContent) {
        console.log('Dynamic content detected, waiting a bit more...');
        await this.sleep(1000);
        break;
      }
      await this.sleep(1000);
    }
  }

  checkPageLoadingStatus() {
    const loadingIndicators = [
      '.loading',
      '.spinner',
      '[aria-label*="Loading"]',
      '[aria-label*="loading"]',
      '.progress',
      '.loading-spinner'
    ];

    const isLoading = loadingIndicators.some(selector => {
      const element = document.querySelector(selector);
      return element && element.offsetParent !== null; // visible
    });

    const hasError = document.body.innerText.toLowerCase().includes('error') ||
                     document.body.innerText.toLowerCase().includes('javascript') ||
                     document.body.innerText.toLowerCase().includes('disabled') ||
                     document.body.innerText.toLowerCase().includes('reload');

    return {
      isLoading,
      hasError,
      readyState: document.readyState,
      contentLength: document.body.innerText.length
    };
  }

  getMainTextContent() {
    // Try Google Chat specific selectors first
    const chatSelectors = [
      '[data-thread-id]', // Google Chat message threads
      '.chat-message',
      '.message-container',
      '[role="main"]',
      '.conversation-container',
      '.messages-list'
    ];

    let chatContent = '';
    for (const selector of chatSelectors) {
      const elements = document.querySelectorAll(selector);
      if (elements.length > 0) {
        chatContent = Array.from(elements)
          .map(el => el.innerText)
          .filter(text => text.trim().length > 0)
          .join('\n\n');
        if (chatContent.trim().length > 50) { // Found substantial chat content
          console.log('Found chat content using selector:', selector);
          break;
        }
      }
    }

    if (chatContent.trim().length > 50) {
      return this.cleanText(chatContent);
    }

    // Fallback to general content extraction
    const clone = document.cloneNode(true);
    const elementsToRemove = clone.querySelectorAll('script, style, nav, header, footer, aside, .advertisement, .ad, .sidebar');
    elementsToRemove.forEach(el => el.remove());

    // Focus on main content areas
    const mainSelectors = ['main', 'article', '.content', '.main-content', '#content', '#main', '[role="main"]'];
    let mainContent = '';

    for (const selector of mainSelectors) {
      const element = clone.querySelector(selector);
      if (element) {
        mainContent = element.innerText;
        break;
      }
    }

    // Fallback to body text if no main content found
    if (!mainContent) {
      mainContent = clone.body?.innerText || '';
    }

    return this.cleanText(mainContent);
  }

  cleanText(text) {
    // Clean up extra whitespace and return meaningful content
    const cleaned = text.replace(/\s+/g, ' ').trim();

    // If text is too short, it might be a loading state
    if (cleaned.length < 20) {
      return 'Page content appears to be loading or very minimal. Consider waiting a moment and trying again.';
    }

    return cleaned;
  }

  extractForms() {
    const forms = [];
    const fields = [];

    // Extract traditional form fields
    document.querySelectorAll('form').forEach((form, index) => {
      const formData = {
        index,
        action: form.action,
        method: form.method,
        fields: []
      };

      form.querySelectorAll('input, textarea, select').forEach(field => {
        if (field.type !== 'hidden') {
          formData.fields.push({
            name: field.name,
            type: field.type,
            placeholder: field.placeholder,
            label: this.getFieldLabel(field),
            required: field.required,
            selector: this.getElementSelector(field),
            isVisible: this.isElementVisible(field)
          });
        }
      });

      forms.push(formData);
    });

    // Extract standalone input fields (not in forms)
    const standaloneFields = [];
    document.querySelectorAll('input:not(form input), textarea:not(form textarea), select:not(form select)').forEach(field => {
      if (field.type !== 'hidden' && this.isElementVisible(field)) {
        standaloneFields.push({
          name: field.name,
          type: field.type,
          placeholder: field.placeholder,
          label: this.getFieldLabel(field),
          required: field.required,
          selector: this.getElementSelector(field),
          isVisible: this.isElementVisible(field)
        });
      }
    });

    if (standaloneFields.length > 0) {
      forms.push({
        index: 'standalone',
        action: '',
        method: '',
        fields: standaloneFields
      });
    }

    // Extract contenteditable elements (rich text editors)
    const editableElements = [];
    document.querySelectorAll('[contenteditable="true"], .ql-editor, .notranslate, [role="textbox"]').forEach(element => {
      if (this.isElementVisible(element)) {
        editableElements.push({
          name: element.getAttribute('data-name') || 'contenteditable',
          type: 'contenteditable',
          placeholder: element.getAttribute('data-placeholder') || 'Rich text editor',
          label: this.getFieldLabel(element) || 'Rich text field',
          required: false,
          selector: this.getElementSelector(element),
          isVisible: this.isElementVisible(element),
          isRichText: true
        });
      }
    });

    if (editableElements.length > 0) {
      forms.push({
        index: 'contenteditable',
        action: '',
        method: '',
        fields: editableElements
      });
    }

    // Special handling for Google Docs, Sheets, etc.
    if (window.location.href.includes('docs.google.com')) {
      const googleDocsEditor = document.querySelector('.kix-canvas-tile-content, .grid-container, [role="textbox"]');
      if (googleDocsEditor) {
        forms.push({
          index: 'google-docs',
          action: '',
          method: '',
          fields: [{
            name: 'document-content',
            type: 'google-docs',
            placeholder: 'Google Docs document',
            label: 'Document content',
            required: false,
            selector: this.getElementSelector(googleDocsEditor),
            isVisible: true,
            isGoogleDocs: true
          }]
        });
      }
    }

    return forms;
  }

  extractLinks() {
    const links = [];
    document.querySelectorAll('a[href]').forEach(link => {
      if (link.href && !link.href.startsWith('javascript:')) {
        links.push({
          text: link.innerText.trim(),
          href: link.href,
          selector: this.getElementSelector(link)
        });
      }
    });
    return links.slice(0, 20); // Limit to first 20 links
  }

  extractButtons() {
    const buttons = [];
    const buttonSelectors = [
      'button',
      'input[type="button"]',
      'input[type="submit"]',
      '[role="button"]',
      '.btn',
      '.button',
      '[data-testid*="button"]',
      '[aria-label*="button"]',
      'a[href="#"]', // Links used as buttons
      'div[onclick]', // Clickable divs
      'span[onclick]', // Clickable spans
      '[tabindex="0"]', // Keyboard accessible elements
      '.clickable'
    ];

    buttonSelectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(button => {
        // Skip if already added
        if (buttons.some(b => b.selector === this.getElementSelector(button))) {
          return;
        }

        const text = this.getElementText(button);
        const isVisible = this.isElementVisible(button);

        if (text.trim() && isVisible) {
          buttons.push({
            text: text.trim(),
            type: button.type || 'button',
            selector: this.getElementSelector(button),
            ariaLabel: button.getAttribute('aria-label') || '',
            title: button.title || '',
            className: button.className || '',
            isInteractive: this.isElementInteractive(button)
          });
        }
      });
    });

    return buttons.sort((a, b) => a.text.length - b.text.length); // Sort by text length
  }

  getElementText(element) {
    // Try different ways to get meaningful text
    return element.innerText ||
           element.textContent ||
           element.value ||
           element.title ||
           element.getAttribute('aria-label') ||
           element.getAttribute('data-tooltip') ||
           element.alt ||
           '';
  }

  isElementVisible(element) {
    const style = window.getComputedStyle(element);
    return style.display !== 'none' &&
           style.visibility !== 'hidden' &&
           style.opacity !== '0' &&
           element.offsetParent !== null;
  }

  isElementInteractive(element) {
    return !element.disabled &&
           !element.readonly &&
           element.tabIndex !== -1 &&
           (element.onclick ||
            element.addEventListener ||
            element.href ||
            ['button', 'input', 'select', 'textarea', 'a'].includes(element.tagName.toLowerCase()) ||
            element.getAttribute('role') === 'button');
  }

  extractImages() {
    const images = [];
    document.querySelectorAll('img').forEach(img => {
      if (img.src && img.alt) {
        images.push({
          src: img.src,
          alt: img.alt,
          title: img.title
        });
      }
    });
    return images.slice(0, 10); // Limit to first 10 images
  }

  extractMetadata() {
    const metadata = {};

    // Meta tags
    document.querySelectorAll('meta[name], meta[property]').forEach(meta => {
      const key = meta.getAttribute('name') || meta.getAttribute('property');
      const content = meta.getAttribute('content');
      if (key && content) {
        metadata[key] = content;
      }
    });

    // Structured data
    const structuredData = [];
    document.querySelectorAll('script[type="application/ld+json"]').forEach(script => {
      try {
        structuredData.push(JSON.parse(script.textContent));
      } catch (e) {
        // Ignore invalid JSON
      }
    });

    if (structuredData.length > 0) {
      metadata.structuredData = structuredData;
    }

    return metadata;
  }

  getFieldLabel(field) {
    // Try to find associated label
    if (field.id) {
      const label = document.querySelector(`label[for="${field.id}"]`);
      if (label) return label.innerText.trim();
    }

    // Try parent label
    const parentLabel = field.closest('label');
    if (parentLabel) return parentLabel.innerText.trim();

    // Try previous sibling
    const prevElement = field.previousElementSibling;
    if (prevElement && (prevElement.tagName === 'LABEL' || prevElement.innerText)) {
      return prevElement.innerText.trim();
    }

    return field.placeholder || field.name || '';
  }

  getElementSelector(element) {
    // Generate a unique selector for the element
    if (element.id) {
      return `#${element.id}`;
    }

    const tagName = element.tagName.toLowerCase();
    let selector = tagName;

    if (element.className) {
      const classes = element.className.split(' ').filter(c => c.length > 0);
      if (classes.length > 0) {
        selector += '.' + classes.join('.');
      }
    }

    // Add position if needed to make it unique
    const siblings = Array.from(element.parentNode?.children || []);
    const index = siblings.indexOf(element);
    if (siblings.filter(s => s.tagName === element.tagName).length > 1) {
      selector += `:nth-child(${index + 1})`;
    }

    return selector;
  }

  async executeTask(instruction) {
    try {
      // Send the instruction and page content to background script for AI processing
      const content = await this.extractPageContent();

      const response = await chrome.runtime.sendMessage({
        action: 'processWithAI',
        instruction,
        content
      });

      if (response.success && response.actions) {
        // Execute the actions returned by AI
        const executionResult = await this.executeActions(response.actions);

        if (executionResult.allSuccessful) {
          return {
            success: true,
            message: response.message || `Task completed successfully! (${executionResult.successCount}/${executionResult.totalActions} actions)`
          };
        } else {
          return {
            success: false,
            error: `Task partially completed: ${executionResult.successCount}/${executionResult.totalActions} actions succeeded. Some actions may not have taken effect visually. Check the page to verify results.`
          };
        }
      } else {
        return { success: false, error: response.error || 'Failed to process instruction' };
      }
    } catch (error) {
      console.error('Error executing task:', error);
      return { success: false, error: error.message };
    }
  }

  async executeActions(actions) {
    let successCount = 0;
    let failCount = 0;
    const results = [];

    for (const action of actions) {
      const result = await this.performAction(action);
      results.push({
        action: action,
        success: result
      });

      if (result) {
        successCount++;
      } else {
        failCount++;
      }

      await this.sleep(500); // Small delay between actions
    }

    return {
      totalActions: actions.length,
      successCount,
      failCount,
      results,
      allSuccessful: failCount === 0
    };
  }

  async performAction(action) {
    console.log(`Performing action: ${action.type} on ${action.selector}`, action);

    switch (action.type) {
      case 'click':
        return await this.clickElementBySelector(action.selector);
      case 'fill':
        return await this.fillField(action.selector, action.value);
      case 'submit':
        return await this.submitForm(action.selector);
      case 'scroll':
        return await this.scrollToElement(action.selector);
      case 'highlight':
        return await this.highlightElement(action.selector);
      default:
        console.error('Unknown action type:', action.type);
        return false;
    }
  }

  async clickElementBySelector(selector) {
    const element = await this.findElement(selector);
    if (!element) {
      console.error('Element not found:', selector);
      return false;
    }

    try {
      // Scroll element into view
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      await this.sleep(500);

      // Try multiple click methods for better compatibility
      const success = await this.tryMultipleClickMethods(element);

      if (success) {
        console.log('Successfully clicked element:', selector);
        return true;
      } else {
        console.error('Failed to click element:', selector);
        return false;
      }
    } catch (error) {
      console.error('Error clicking element:', error);
      return false;
    }
  }

  async tryMultipleClickMethods(element) {
    const methods = [
      // Method 1: Standard click
      () => {
        element.click();
        return true;
      },

      // Method 2: Mouse events
      () => {
        const rect = element.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        ['mousedown', 'mouseup', 'click'].forEach(eventType => {
          const event = new MouseEvent(eventType, {
            bubbles: true,
            cancelable: true,
            clientX: centerX,
            clientY: centerY,
            button: 0
          });
          element.dispatchEvent(event);
        });
        return true;
      },

      // Method 3: Focus and keyboard
      () => {
        element.focus();
        const enterEvent = new KeyboardEvent('keydown', {
          key: 'Enter',
          keyCode: 13,
          bubbles: true
        });
        element.dispatchEvent(enterEvent);
        return true;
      },

      // Method 4: Touch events (for mobile-optimized sites)
      () => {
        const touchEvent = new TouchEvent('touchstart', {
          bubbles: true,
          touches: [new Touch({
            identifier: 1,
            target: element,
            clientX: element.getBoundingClientRect().left + 10,
            clientY: element.getBoundingClientRect().top + 10
          })]
        });
        element.dispatchEvent(touchEvent);
        element.click();
        return true;
      }
    ];

    for (const method of methods) {
      try {
        method();
        await this.sleep(200);
        return true;
      } catch (error) {
        console.log('Click method failed, trying next:', error.message);
      }
    }

    return false;
  }

  async fillField(selector, value) {
    const element = await this.findElement(selector);
    if (!element) {
      console.error('Field not found:', selector);
      return false;
    }

    try {
      // Handle different types of input elements
      const success = await this.tryMultipleFillMethods(element, value);

      if (success) {
        console.log('Successfully filled field:', selector, 'with:', value);
        return true;
      } else {
        console.error('Failed to fill field:', selector);
        return false;
      }
    } catch (error) {
      console.error('Error filling field:', error);
      return false;
    }
  }

  async tryMultipleFillMethods(element, value) {
    const methods = [
      // Method 1: Standard form input
      () => {
        element.focus();
        element.select?.(); // Select existing text if possible
        element.value = value;

        // Dispatch events that modern frameworks expect
        ['input', 'change', 'blur'].forEach(eventType => {
          const event = new Event(eventType, { bubbles: true });
          element.dispatchEvent(event);
        });
        return true;
      },

      // Method 2: Rich text editors and contenteditable
      () => {
        if (element.contentEditable === 'true' || element.isContentEditable) {
          element.focus();

          // Clear existing content
          if (document.execCommand) {
            document.execCommand('selectAll');
            document.execCommand('delete');
            document.execCommand('insertText', false, value);
          } else {
            // Fallback for newer browsers
            element.innerText = value;
          }

          // Trigger input events
          const inputEvent = new InputEvent('input', {
            bubbles: true,
            inputType: 'insertText',
            data: value
          });
          element.dispatchEvent(inputEvent);
          return true;
        }
        return false;
      },

      // Method 3: Keyboard simulation for complex editors
      () => {
        element.focus();

        // Clear existing content with Ctrl+A, Delete
        ['keydown', 'keyup'].forEach(eventType => {
          element.dispatchEvent(new KeyboardEvent(eventType, {
            key: 'a',
            ctrlKey: true,
            bubbles: true
          }));
        });

        // Type the new value character by character
        for (const char of value) {
          ['keydown', 'keypress', 'input', 'keyup'].forEach(eventType => {
            const event = eventType === 'input'
              ? new InputEvent(eventType, { bubbles: true, data: char })
              : new KeyboardEvent(eventType, { key: char, bubbles: true });
            element.dispatchEvent(event);
          });
        }
        return true;
      },

      // Method 4: Advanced Google Docs/Sheets methods
      () => {
        if (window.location.href.includes('docs.google.com') ||
            window.location.href.includes('sheets.google.com')) {

          element.focus();

          // Method 4a: Try clipboard approach
          try {
            // Clear selection first
            const selection = window.getSelection();
            selection.removeAllRanges();

            // Select the element or create a range
            const range = document.createRange();
            range.selectNodeContents(element);
            selection.addRange(range);

            // Try to use clipboard API
            navigator.clipboard.writeText(value).then(() => {
              document.execCommand('paste');
            }).catch(() => {
              // Fallback to direct manipulation
              this.directGoogleDocsInsertion(element, value);
            });

            return true;
          } catch (e) {
            return this.directGoogleDocsInsertion(element, value);
          }
        }
        return false;
      },

      // Method 5: Direct DOM manipulation for stubborn editors
      () => {
        element.focus();

        // Try to simulate typing by manipulating the DOM directly
        if (element.contentEditable === 'true' || element.isContentEditable) {
          // Clear content
          element.innerHTML = '';

          // Create text node and insert
          const textNode = document.createTextNode(value);
          element.appendChild(textNode);

          // Place cursor at end
          const selection = window.getSelection();
          const range = document.createRange();
          range.setStartAfter(textNode);
          range.collapse(true);
          selection.removeAllRanges();
          selection.addRange(range);

          // Fire events to notify the application
          ['input', 'textInput', 'change'].forEach(eventType => {
            const event = new Event(eventType, { bubbles: true });
            element.dispatchEvent(event);
          });

          return true;
        }
        return false;
      },

      // Method 6: Character-by-character keyboard simulation
      () => {
        element.focus();

        // Use more realistic typing simulation
        for (let i = 0; i < value.length; i++) {
          const char = value[i];

          // Simulate keydown, keypress, input, keyup for each character
          const keydownEvent = new KeyboardEvent('keydown', {
            key: char,
            char: char,
            charCode: char.charCodeAt(0),
            keyCode: char.charCodeAt(0),
            which: char.charCodeAt(0),
            bubbles: true
          });

          const keypressEvent = new KeyboardEvent('keypress', {
            key: char,
            char: char,
            charCode: char.charCodeAt(0),
            keyCode: char.charCodeAt(0),
            which: char.charCodeAt(0),
            bubbles: true
          });

          const inputEvent = new InputEvent('input', {
            bubbles: true,
            inputType: 'insertText',
            data: char
          });

          const keyupEvent = new KeyboardEvent('keyup', {
            key: char,
            char: char,
            charCode: char.charCodeAt(0),
            keyCode: char.charCodeAt(0),
            which: char.charCodeAt(0),
            bubbles: true
          });

          element.dispatchEvent(keydownEvent);
          element.dispatchEvent(keypressEvent);
          element.dispatchEvent(inputEvent);
          element.dispatchEvent(keyupEvent);
        }

        return true;
      }
    ];

    for (let i = 0; i < methods.length; i++) {
      try {
        console.log(`Trying fill method ${i + 1}/${methods.length}`);
        const result = methods[i]();
        if (result) {
          await this.sleep(500);

          // Verify the text was actually inserted
          const success = await this.verifyTextInsertion(element, value);
          if (success) {
            console.log(`Method ${i + 1} succeeded and verified`);
            return true;
          } else {
            console.log(`Method ${i + 1} reported success but verification failed`);
          }
        }
      } catch (error) {
        console.log(`Fill method ${i + 1} failed:`, error.message);
      }
    }

    return false;
  }

  async verifyTextInsertion(element, expectedValue) {
    // Wait a moment for the change to take effect
    await this.sleep(300);

    // Check various ways the text might be stored
    const actualContent =
      element.value ||
      element.innerText ||
      element.textContent ||
      '';

    const isPresent = actualContent.includes(expectedValue) ||
                     actualContent.toLowerCase().includes(expectedValue.toLowerCase());

    console.log('Verification:', {
      expected: expectedValue,
      actual: actualContent.substring(0, 100),
      isPresent: isPresent
    });

    return isPresent;
  }

  directGoogleDocsInsertion(element, value) {
    try {
      // For Google Docs, try to find the actual editable area
      const editableArea = document.querySelector('.kix-canvas-tile-content') ||
                          document.querySelector('.docs-texteventtarget-iframe') ||
                          document.querySelector('[role="textbox"]') ||
                          element;

      if (editableArea !== element) {
        editableArea.focus();
        element = editableArea;
      }

      // Try the nuclear option - direct text insertion
      if (document.execCommand) {
        // Select all and delete first
        document.execCommand('selectAll');
        document.execCommand('delete');

        // Insert the new text
        const success = document.execCommand('insertText', false, value);
        if (success) return true;
      }

      // Last resort - try to find and manipulate the underlying data
      if (window.gapi && window.gapi.drive) {
        // This would require Google Drive API access
        console.log('Google Drive API available but not implemented');
      }

      return false;
    } catch (error) {
      console.error('directGoogleDocsInsertion failed:', error);
      return false;
    }
  }

  async findElement(selector) {
    // Try direct selector first
    let element = document.querySelector(selector);
    if (element) return element;

    // Wait and try again for dynamic content
    await this.sleep(1000);
    element = document.querySelector(selector);
    if (element) return element;

    // Try to find by similar attributes
    const selectorParts = selector.split(/[.#\[\]]/);
    for (const part of selectorParts) {
      if (part.trim()) {
        element = document.querySelector(`[class*="${part}"]`) ||
                 document.querySelector(`[id*="${part}"]`) ||
                 document.querySelector(`[data-*="${part}"]`);
        if (element) {
          console.log('Found element using partial match:', part);
          return element;
        }
      }
    }

    return null;
  }

  fillFormFields(fields) {
    fields.forEach(field => {
      this.fillField(field.selector, field.value);
    });
  }

  submitForm(selector) {
    const form = document.querySelector(selector);
    if (form && form.tagName === 'FORM') {
      form.submit();
    }
  }

  scrollToElement(selector) {
    const element = document.querySelector(selector);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  highlightElement(selector) {
    const element = document.querySelector(selector);
    if (element) {
      element.style.outline = '3px solid #4285f4';
      element.style.backgroundColor = 'rgba(66, 133, 244, 0.1)';
      this.highlightedElements.add(element);
    }
  }

  highlightInteractableElements() {
    const selectors = [
      'button',
      'input[type="button"]',
      'input[type="submit"]',
      'a[href]',
      'input[type="text"]',
      'input[type="email"]',
      'input[type="password"]',
      'textarea',
      'select',
      '[role="button"]',
      '[onclick]'
    ];

    selectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(element => {
        this.highlightElement(this.getElementSelector(element));
      });
    });
  }

  clearHighlights() {
    this.highlightedElements.forEach(element => {
      element.style.outline = '';
      element.style.backgroundColor = '';
    });
    this.highlightedElements.clear();
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize the assistant
const webAssistant = new WebAssistant();

// Inject styles for highlighting
const style = document.createElement('style');
style.textContent = `
  .ai-assistant-highlight {
    outline: 3px solid #4285f4 !important;
    background-color: rgba(66, 133, 244, 0.1) !important;
    transition: all 0.3s ease !important;
  }
`;
document.head.appendChild(style);