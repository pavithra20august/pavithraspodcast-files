import json
import os
import random
import re
from collections import defaultdict, Counter
from datetime import datetime
import pickle

class SelfLearningChatbot:
    def __init__(self, knowledge_file="chatbot_knowledge.pkl"):
        self.knowledge_file = knowledge_file
        
        # Core learning components
        self.response_patterns = defaultdict(list)  # input pattern -> responses
        self.word_associations = defaultdict(Counter)  # word -> related words
        self.conversation_history = []
        self.response_ratings = defaultdict(list)  # track response quality
        
        # Load existing knowledge
        self.load_knowledge()
        
        # Initialize with some basic responses
        self._initialize_basic_responses()
    
    def _initialize_basic_responses(self):
        """Start with basic responses if no knowledge exists"""
        if not self.response_patterns:
            basic_patterns = {
                "hello": ["Hello! How are you?", "Hi there!", "Hey! Nice to see you!"],
                "how are you": ["I'm doing well, thanks for asking!", "I'm good! How about you?"],
                "bye": ["Goodbye! Have a great day!", "See you later!", "Take care!"],
                "thanks": ["You're welcome!", "Happy to help!", "No problem!"],
                "what": ["That's an interesting question!", "Let me think about that..."],
                "default": ["That's interesting!", "Tell me more!", "I see!", "Go on..."]
            }
            
            for pattern, responses in basic_patterns.items():
                self.response_patterns[pattern] = responses
    
    def _extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        return [word for word in words if word not in stop_words]
    
    def _find_best_pattern(self, user_input):
        """Find the best matching pattern for user input"""
        keywords = self._extract_keywords(user_input)
        user_input_lower = user_input.lower()
        
        # Direct keyword matching
        for keyword in keywords:
            if keyword in self.response_patterns:
                return keyword
        
        # Partial matching
        for pattern in self.response_patterns:
            if pattern in user_input_lower:
                return pattern
        
        # Check for greeting patterns
        greetings = ['hello', 'hi', 'hey', 'greetings']
        if any(greeting in user_input_lower for greeting in greetings):
            return 'hello'
        
        # Check for farewell patterns
        farewells = ['bye', 'goodbye', 'see you', 'farewell']
        if any(farewell in user_input_lower for farewell in farewells):
            return 'bye'
        
        # Check for questions
        if any(word in user_input_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return 'what'
        
        return 'default'
    
    def generate_response(self, user_input):
        """Generate a response based on learned patterns"""
        pattern = self._find_best_pattern(user_input)
        
        if pattern in self.response_patterns and self.response_patterns[pattern]:
            # Choose response based on past ratings (if available)
            responses = self.response_patterns[pattern]
            
            # Weighted selection based on past performance
            if pattern in self.response_ratings:
                ratings = self.response_ratings[pattern]
                if len(ratings) >= len(responses):
                    # Use ratings to weight selection
                    weights = []
                    for i, response in enumerate(responses):
                        avg_rating = sum(ratings[i::len(responses)]) / max(1, len(ratings[i::len(responses)]))
                        weights.append(max(0.1, avg_rating))  # Minimum weight of 0.1
                    
                    # Weighted random selection
                    response = random.choices(responses, weights=weights)[0]
                else:
                    response = random.choice(responses)
            else:
                response = random.choice(responses)
        else:
            response = "I'm still learning! Could you tell me more?"
        
        return response, pattern
    
    def learn_from_conversation(self, user_input, bot_response, pattern, user_feedback=None):
        """Learn from the conversation"""
        # Store conversation
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'pattern': pattern
        }
        self.conversation_history.append(conversation_entry)
        
        # Learn word associations
        user_keywords = self._extract_keywords(user_input)
        response_keywords = self._extract_keywords(bot_response)
        
        for user_word in user_keywords:
            for response_word in response_keywords:
                self.word_associations[user_word][response_word] += 1
        
        # If user provides feedback, use it for learning
        if user_feedback is not None:
            self.response_ratings[pattern].append(user_feedback)
        
        # Auto-learn new patterns from user input
        self._auto_learn_patterns(user_input, user_keywords)
    
    def _auto_learn_patterns(self, user_input, keywords):
        """Automatically learn new response patterns"""
        # If this is a new pattern, try to generate a response
        for keyword in keywords:
            if keyword not in self.response_patterns:
                # Create new pattern with generic responses
                self.response_patterns[keyword] = [
                    f"Tell me more about {keyword}!",
                    f"That's interesting about {keyword}.",
                    f"I'd like to learn more about {keyword}."
                ]
    
    def add_custom_response(self, pattern, response):
        """Allow manual addition of responses"""
        pattern = pattern.lower()
        self.response_patterns[pattern].append(response)
        print(f"Added response for pattern '{pattern}': {response}")
    
    def save_knowledge(self):
        """Save learned knowledge to file"""
        knowledge = {
            'response_patterns': dict(self.response_patterns),
            'word_associations': {k: dict(v) for k, v in self.word_associations.items()},
            'conversation_history': self.conversation_history,
            'response_ratings': dict(self.response_ratings)
        }
        
        with open(self.knowledge_file, 'wb') as f:
            pickle.dump(knowledge, f)
        print(f"Knowledge saved to {self.knowledge_file}")
    
    def load_knowledge(self):
        """Load previously learned knowledge"""
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'rb') as f:
                    knowledge = pickle.load(f)
                
                self.response_patterns = defaultdict(list, knowledge.get('response_patterns', {}))
                self.word_associations = defaultdict(Counter, 
                    {k: Counter(v) for k, v in knowledge.get('word_associations', {}).items()})
                self.conversation_history = knowledge.get('conversation_history', [])
                self.response_ratings = defaultdict(list, knowledge.get('response_ratings', {}))
                
                print(f"Loaded knowledge from {self.knowledge_file}")
                print(f"I know {len(self.response_patterns)} patterns and had {len(self.conversation_history)} conversations!")
            except:
                print("Could not load previous knowledge, starting fresh!")
    
    def get_stats(self):
        """Get chatbot learning statistics"""
        return {
            'patterns_learned': len(self.response_patterns),
            'total_conversations': len(self.conversation_history),
            'vocabulary_size': len(self.word_associations),
            'total_responses': sum(len(responses) for responses in self.response_patterns.values())
        }

def main():
    """Main chat interface"""
    print("🤖 Self-Learning Chatbot Started!")
    print("Type 'quit' to exit, 'stats' for learning statistics, 'teach' to add custom responses")
    print("After each response, you can rate it 1-5 (5 being best) to help me learn!")
    print("-" * 60)
    
    bot = SelfLearningChatbot()
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Bot: Goodbye! Thanks for helping me learn!")
                bot.save_knowledge()
                break
            
            elif user_input.lower() == 'stats':
                stats = bot.get_stats()
                print("\n📊 Learning Statistics:")
                for key, value in stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                continue
            
            elif user_input.lower() == 'teach':
                pattern = input("Enter the pattern/keyword: ").strip()
                response = input("Enter the response: ").strip()
                bot.add_custom_response(pattern, response)
                continue
            
            elif not user_input:
                continue
            
            # Generate response
            response, pattern = bot.generate_response(user_input)
            print(f"Bot: {response}")
            
            # Get user feedback for learning
            try:
                feedback = input("Rate this response (1-5, or press Enter to skip): ").strip()
                if feedback and feedback.isdigit():
                    feedback_score = int(feedback)
                    if 1 <= feedback_score <= 5:
                        bot.learn_from_conversation(user_input, response, pattern, feedback_score)
                        print(f"Thanks! I'll remember that this response was rated {feedback_score}/5")
                    else:
                        bot.learn_from_conversation(user_input, response, pattern)
                else:
                    bot.learn_from_conversation(user_input, response, pattern)
            except:
                bot.learn_from_conversation(user_input, response, pattern)
    
    except KeyboardInterrupt:
        print("\n\nBot: Thanks for chatting! Saving what I learned...")
        bot.save_knowledge()

if __name__ == "__main__":
    main()