from template.agent import BaseAgent
from template.agent.meta.state import AgentState
from template.message.message import HumanMessage, SystemMessage
from template.message.converter import convert_messages_list
from template.agent.meta.utils import extract_from_xml
from template.agent.meta.prompt import META_PROMPT
from autogen import AssistantAgent, UserProxyAgent
from template.agent.autogen_config import create_autogen_agent, create_user_proxy_agent

from termcolor import colored
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

class MetaAgent:
    def __init__(self,
                 tools: list = [],
                 model: str = "gemini-2.5-pro",
                 temperature: float = 0.2,
                 max_iteration: int = 10,
                 json_mode: bool = False,
                 verbose: bool = False):
        self.name = "Meta Agent"
        self.model = model
        self.temperature = temperature
        self.tools = tools
        self.max_iteration = max_iteration
        self.json_mode = json_mode
        self.verbose = verbose
        self.iteration = 0

        # Create AutoGen agent using the new configuration
        self.agent = create_autogen_agent(
            name=self.name,
            system_message=META_PROMPT,
            model=model,
            temperature=temperature,
            tools=tools
        )

    def invoke(self, input_data) -> dict:
        """Main invoke method for MetaAgent"""
        if self.verbose:
            print(f'Entering ' + colored(self.name, 'black', 'on_white'))

        # Handle different input formats
        if isinstance(input_data, dict):
            # Convert dict to formatted string for MetaAgent
            task = input_data.get('task', '')
            context = input_data.get('context', '')
            previous_results = input_data.get('previous_results', [])

            input_text = f"Task: {task}\nContext: {context}"
            if previous_results:
                input_text += f"\nPrevious Results: {previous_results}"
        else:
            input_text = str(input_data)

        try:
            # Use AutoGen to process the meta-analysis
            user_proxy = UserProxyAgent(
                name="user_proxy",
                code_execution_config=False,
                human_input_mode="NEVER"
            )

            chat_result = user_proxy.initiate_chat(
                self.agent,
                message=f"Analyze this task: {input_text}",
                max_turns=1
            )

            # Extract response
            if chat_result and chat_result.chat_history:
                response_content = chat_result.chat_history[-1].get('content', '')

                # Try to extract structured data from XML-like response
                agent_data = extract_from_xml(response_content)

                if self.verbose:
                    if agent_data:
                        name = agent_data.get('Agent Name', 'Unknown')
                        description = agent_data.get('Agent Description', '')
                        tasks = agent_data.get('Tasks', '')
                        tool = agent_data.get('Tool', '')
                        answer = agent_data.get('Answer', '')

                        if not answer:
                            content = f'Agent Name: {name}\nDescription: {description}\nTasks: {tasks}\nTool: {tool}'
                            print_stmt = colored(content, color='yellow', attrs=['bold'])
                        else:
                            content = f'Final Answer: {answer}'
                            print_stmt = colored(content, color='cyan', attrs=['bold'])
                        print(print_stmt)
                    else:
                        logger.warning("Could not extract structured data from response")

                # Return dict format for compatibility
                return {
                    'output': response_content,
                    'agent_data': agent_data or {},
                    'success': True
                }
            else:
                return {
                    'output': 'No response from MetaAgent',
                    'agent_data': {},
                    'success': False
                }

        except Exception as e:
            logger.error(f"Error in MetaAgent: {str(e)}")
            return {
                'output': f'Error in MetaAgent: {str(e)}',
                'agent_data': {},
                'success': False
            }

    def stream(self, input: str):
        """Placeholder for streaming"""
        pass