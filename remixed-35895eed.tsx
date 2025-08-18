import React, { useState } from 'react';
import { ChevronRight, Bot, Settings, BarChart3, GitBranch, Play, Key, Eye, EyeOff } from 'lucide-react';

const LangChainExplorer = () => {
  const [activeTab, setActiveTab] = useState('langchain');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState('');

  const tools = {
    langchain: {
      icon: <Bot className="w-8 h-8 text-blue-500" />,
      title: "LangChain",
      subtitle: "The Foundation Framework",
      description: "LangChain is like a toolkit for building AI applications. Think of it as LEGO blocks for AI - you can connect different AI services, databases, and tools together.",
      example: "Building a Customer Service Bot",
      code: `from langchain.llms import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up the AI model
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Create a prompt template
template = """
You are a helpful customer service representative for TechStore.
Customer question: {question}
Respond professionally and helpfully.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

# Use the chain
response = chain.run(question="How do I return a product?")`,
      realWorld: "Like having a smart assistant that can handle customer questions, process orders, and provide support 24/7."
    },
    langflow: {
      icon: <Settings className="w-8 h-8 text-green-500" />,
      title: "LangFlow",
      subtitle: "Visual AI Builder",
      description: "LangFlow is like a drag-and-drop website builder, but for AI applications. Instead of writing code, you connect visual blocks to create AI workflows.",
      example: "Visual Customer Service Flow",
      code: `# LangFlow creates this code automatically from your visual flow:

# Input Node: Customer Message
customer_input = TextInput(name="customer_question")

# Processing Node: Intent Classification
intent_classifier = LLMNode(
    model="gemini-pro",
    prompt="Classify this customer message: {input}\\nCategories: billing, returns, technical"
)

# Decision Node: Route Based on Intent
router = ConditionalRouter(
    conditions={
        "billing": billing_chain,
        "returns": returns_chain, 
        "technical": tech_support_chain
    }
)

# Output Node: Send Response
response_sender = OutputNode()

# Visual Flow: Input → Classify → Route → Respond`,
      realWorld: "Like creating a flowchart that actually works - perfect for non-programmers who want to build AI apps."
    },
    langsmith: {
      icon: <BarChart3 className="w-8 h-8 text-purple-500" />,
      title: "LangSmith",
      subtitle: "AI Performance Monitor",
      description: "LangSmith is like Google Analytics for your AI applications. It tracks how well your AI is performing and helps you improve it over time.",
      example: "Monitoring Customer Satisfaction",
      code: `from langsmith import Client

client = Client(api_key="your-langsmith-key")

# Track every customer interaction
@client.trace
def handle_customer_query(question):
    response = customer_service_chain.run(question)
    
    # Log the interaction for analysis
    client.log_run(
        name="customer_service",
        inputs={"question": question},
        outputs={"response": response},
        metadata={
            "customer_id": "12345",
            "timestamp": datetime.now(),
            "satisfaction_score": None  # Will be filled later
        }
    )
    return response

# Analyze performance
metrics = client.get_run_metrics(
    project="customer_service",
    filters={"date_range": "last_week"}
)`,
      realWorld: "Like having a dashboard that shows which customer questions your AI handles well and which ones need improvement."
    },
    langgraph: {
      icon: <GitBranch className="w-8 h-8 text-orange-500" />,
      title: "LangGraph", 
      subtitle: "Smart Decision Maker",
      description: "LangGraph creates AI agents that can make decisions, use tools, and handle complex multi-step tasks. Think of it as giving your AI a brain that can think through problems step by step.",
      example: "Smart Customer Service Agent",
      code: `from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutor

# Define what the agent can remember
class CustomerState(TypedDict):
    question: str
    customer_info: dict
    tools_used: list
    response: str

# Create tools the agent can use
tools = [
    check_order_status,
    process_refund,
    escalate_to_human
]

# Build the decision graph
workflow = StateGraph(CustomerState)

# Add decision points
workflow.add_node("analyze_question", analyze_customer_question)
workflow.add_node("check_account", lookup_customer_info) 
workflow.add_node("use_tools", ToolExecutor(tools))
workflow.add_node("generate_response", create_response)

# Define the decision flow
workflow.add_edge("analyze_question", "check_account")
workflow.add_conditional_edges(
    "check_account",
    should_use_tools,
    {"yes": "use_tools", "no": "generate_response"}
)

# The agent thinks through each step
app = workflow.compile()`,
      realWorld: "Like having an AI employee that can look up your order, check policies, process refunds, or get help from humans when needed."
    }
  };

  const handleRunExample = () => {
    if (!apiKey.trim()) {
      setOutput('❌ Please enter your Gemini API key first!');
      return;
    }

    setIsRunning(true);
    setOutput('🚀 Running example...\n');

    // Simulate the execution with realistic delays
    setTimeout(() => {
      const tool = tools[activeTab];
      let simulatedOutput = '';

      switch (activeTab) {
        case 'langchain':
          simulatedOutput = `✅ LangChain Example Complete!

🤖 Customer Question: "How do I return a product?"

📝 AI Response: "Hello! I'd be happy to help you with your return. Here's our simple return process:

1. Visit our returns portal within 30 days of purchase
2. Select your order and reason for return  
3. Print the prepaid return label
4. Package the item securely
5. Drop it off at any shipping location

Your refund will be processed within 3-5 business days once we receive the item. Is there anything specific about your return I can help clarify?"

💡 This took 2.3 seconds and cost $0.002`;
          break;

        case 'langflow':
          simulatedOutput = `✅ LangFlow Visual Workflow Complete!

🔄 Flow Execution Trace:
1. 📨 Customer Input: "My order is late and I'm frustrated"
2. 🎯 Intent Classifier: Detected "shipping_inquiry" + "negative_sentiment" 
3. 🔀 Router: Directed to "shipping_support_with_empathy" flow
4. 📋 Order Lookup: Found order #12345, shipped 2 days ago
5. 💬 Response Generator: Crafted empathetic response with tracking info

🤖 Final Response: "I completely understand your frustration, and I sincerely apologize for the delay. I've checked your order #12345, and I can see it's currently in transit. Here's your tracking number: 1Z999... It should arrive by tomorrow evening. As an apology, I'll apply a 10% discount to your next order."

⚡ Visual flow made this complex logic easy to build!`;
          break;

        case 'langsmith':
          simulatedOutput = `✅ LangSmith Monitoring Dashboard Updated!

📊 Performance Metrics (Last 24 hours):
• Total Interactions: 147
• Average Response Time: 1.8s
• Customer Satisfaction: 4.2/5 ⭐
• Resolution Rate: 89%

🎯 Top Performing Responses:
1. Return/Exchange queries (95% satisfaction)
2. Product information (92% satisfaction) 
3. Order status checks (88% satisfaction)

⚠️ Areas for Improvement:
• Technical support queries (67% satisfaction)
• Billing disputes (71% satisfaction)

🔧 Recommendations:
- Add more technical knowledge to training data
- Create specialized billing dispute workflow
- Consider human handoff for complex technical issues

📈 Trend: Customer satisfaction improved 12% this week!`;
          break;

        case 'langgraph':
          simulatedOutput = `✅ LangGraph Agent Execution Complete!

🧠 Agent Decision Process:
1. 📨 Received: "I want to return my laptop but lost the receipt"
2. 🔍 Analyzed: Return request + missing documentation
3. 👤 Looked up customer: John Smith, Premium member since 2020
4. 🎯 Decision: Check purchase history tool
5. 💾 Found: MacBook Pro purchased 15 days ago, order #54321
6. 📋 Policy check: Premium members get receipt-free returns
7. 🔧 Action: Generated return label automatically

🤖 Agent Response: "Good news, John! As a Premium member, you don't need your receipt. I found your MacBook Pro purchase from 15 days ago. I've generated a return label and emailed it to you. The return window is extended to 60 days for Premium members. Just drop it off at any of our locations!"

🛠️ Tools Used:
- Customer lookup ✓
- Purchase history search ✓  
- Policy database query ✓
- Return label generator ✓

⚡ The agent handled this complex scenario automatically!`;
          break;
      }

      setOutput(simulatedOutput);
      setIsRunning(false);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
            LangChain Ecosystem Explorer
          </h1>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto">
            Learn about LangChain's powerful tools through a practical customer service chatbot example
          </p>
        </div>

        {/* API Key Input */}
        <div className="mb-8 max-w-2xl mx-auto">
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
            <div className="flex items-center mb-3">
              <Key className="w-5 h-5 text-yellow-400 mr-2" />
              <h3 className="text-lg font-semibold text-yellow-400">Gemini API Key</h3>
            </div>
            <div className="relative">
              <input
                type={showApiKey ? "text" : "password"}
                placeholder="Enter your Gemini API key to run examples..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 pr-12"
              />
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-white"
              >
                {showApiKey ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
            <p className="text-sm text-slate-400 mt-2">
              🔒 Your API key is stored locally and never sent to any server except Google's Gemini API
            </p>
          </div>
        </div>

        {/* Tool Navigation */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          {Object.entries(tools).map(([key, tool]) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`flex items-center space-x-3 px-6 py-4 rounded-xl transition-all duration-300 ${
                activeTab === key
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 shadow-lg shadow-blue-500/25 scale-105'
                  : 'bg-slate-800/50 border border-slate-700 hover:bg-slate-700/50 hover:border-slate-600'
              }`}
            >
              {tool.icon}
              <div className="text-left">
                <div className="font-semibold">{tool.title}</div>
                <div className="text-sm text-slate-400">{tool.subtitle}</div>
              </div>
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Panel - Explanation */}
            <div className="space-y-6">
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
                <div className="flex items-center mb-4">
                  {tools[activeTab].icon}
                  <div className="ml-3">
                    <h2 className="text-2xl font-bold">{tools[activeTab].title}</h2>
                    <p className="text-slate-400">{tools[activeTab].subtitle}</p>
                  </div>
                </div>
                <p className="text-slate-300 text-lg leading-relaxed mb-4">
                  {tools[activeTab].description}
                </p>
                <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg p-4 border border-blue-500/20">
                  <h3 className="font-semibold text-blue-300 mb-2">Real-World Application:</h3>
                  <p className="text-slate-300">{tools[activeTab].realWorld}</p>
                </div>
              </div>

              {/* Example Scenario */}
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
                <h3 className="text-xl font-semibold mb-4 text-green-400">
                  📚 {tools[activeTab].example}
                </h3>
                <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
                  <pre className="text-sm text-slate-300 overflow-x-auto whitespace-pre-wrap">
                    <code>{tools[activeTab].code}</code>
                  </pre>
                </div>
              </div>
            </div>

            {/* Right Panel - Interactive Demo */}
            <div className="space-y-6">
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold">🚀 Try It Live</h3>
                  <button
                    onClick={handleRunExample}
                    disabled={isRunning || !apiKey.trim()}
                    className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-all ${
                      isRunning || !apiKey.trim()
                        ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white shadow-lg hover:shadow-green-500/25'
                    }`}
                  >
                    <Play className="w-4 h-4" />
                    <span>{isRunning ? 'Running...' : 'Run Example'}</span>
                  </button>
                </div>
                
                <div className="bg-slate-900 rounded-lg p-4 min-h-[400px] border border-slate-600">
                  <div className="flex items-center mb-3">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <span className="ml-4 text-sm text-slate-400">Output Console</span>
                  </div>
                  <div className="text-sm font-mono">
                    {output ? (
                      <pre className="text-green-400 whitespace-pre-wrap">{output}</pre>
                    ) : (
                      <div className="text-slate-500 italic">
                        Click "Run Example" to see {tools[activeTab].title} in action...
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Key Benefits */}
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
                <h3 className="text-xl font-semibold mb-4 text-purple-400">✨ Key Benefits</h3>
                <div className="space-y-3">
                  {activeTab === 'langchain' && (
                    <>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-blue-400 mt-0.5" />
                        <span className="text-slate-300">Connect any AI model with any data source</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-blue-400 mt-0.5" />
                        <span className="text-slate-300">Reusable components for faster development</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-blue-400 mt-0.5" />
                        <span className="text-slate-300">Large community and extensive documentation</span>
                      </div>
                    </>
                  )}
                  {activeTab === 'langflow' && (
                    <>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-green-400 mt-0.5" />
                        <span className="text-slate-300">No coding required - drag and drop interface</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-green-400 mt-0.5" />
                        <span className="text-slate-300">Visual debugging and testing</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-green-400 mt-0.5" />
                        <span className="text-slate-300">Perfect for prototyping and non-technical users</span>
                      </div>
                    </>
                  )}
                  {activeTab === 'langsmith' && (
                    <>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-purple-400 mt-0.5" />
                        <span className="text-slate-300">Track performance and costs in real-time</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-purple-400 mt-0.5" />
                        <span className="text-slate-300">Identify and fix problems quickly</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-purple-400 mt-0.5" />
                        <span className="text-slate-300">A/B test different AI models and prompts</span>
                      </div>
                    </>
                  )}
                  {activeTab === 'langgraph' && (
                    <>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-orange-400 mt-0.5" />
                        <span className="text-slate-300">AI agents that can use tools and make decisions</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-orange-400 mt-0.5" />
                        <span className="text-slate-300">Handle complex multi-step workflows</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <ChevronRight className="w-5 h-5 text-orange-400 mt-0.5" />
                        <span className="text-slate-300">Memory and context across interactions</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 pt-8 border-t border-slate-700">
          <p className="text-slate-400">
            🎯 Perfect for beginners: Start with LangChain → Visualize with LangFlow → Monitor with LangSmith → Scale with LangGraph
          </p>
        </div>
      </div>
    </div>
  );
};

export default LangChainExplorer;