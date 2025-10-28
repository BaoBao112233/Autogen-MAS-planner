from template.agent import AutoGenAgent, create_vertex_llm_config
from template.agent import BaseAgent

from template.agent.plan.state import PlanState, UpdateState

from template.message.message import SystemMessage, HumanMessage

from template.message.converter import convert_messages_list
from template.message.message import SystemMessage,HumanMessage

from template.agent.plan.utils import (
    read_markdown_file,
    extract_llm_response, 
    extract_priority_plans
)

from template.agent.plan.prompts import PLAN_PROMPTS, UPDATE_PLAN_PROMPTS

from template.agent.meta import MetaAgentfrom template.agent.plan.prompts import PLAN_PROMPTS, UPDATE_PLAN_PROMPTS

from template.agent.tool import ToolAgentfrom template.agent.meta import MetaAgent

from template.agent.api_client import APIClientfrom template.agent.tool import ToolAgent

from template.configs.environments import envfrom template.agent.api_client import APIClient

from autogen import AssistantAgent, UserProxyAgentfrom template.configs.environments import env



from termcolor import coloredfrom langchain_google_vertexai import ChatVertexAI

import osfrom langgraph.graph import StateGraph,END,START

import timefrom langchain_mcp_adapters.client import MultiServerMCPClient

import asynciofrom termcolor import colored

import loggingimport os

from typing import Dict, Anyimport time

import asyncio

# Configure loggingimport logging

logging.basicConfig(

    level=logging.INFO,# Configure logging

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',logging.basicConfig(

    datefmt='%Y-%m-%d %H:%M:%S',    level=logging.INFO,

)    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

logger = logging.getLogger(__name__)    datefmt='%Y-%m-%d %H:%M:%S',  # format thời gian

)

class PlanAgent(AutoGenAgent):logger = logging.getLogger(__name__)

    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.2, max_iteration=10, verbose=True):

        super().__init__(class PlanAgent(BaseAgent):

            name="Plan Agent",    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.2, max_iteration=10,verbose=True):

            system_message="You are a Plan Agent that creates smart home automation plans.",        super().__init__()

            model=model,        

            temperature=temperature        self.name = "Plan Agent"

        )

        self.model = model

        self.model = model        self.temperature = temperature

        self.temperature = temperature        self.verbose = verbose

        self.verbose = verbose

        self.max_iteration = max_iteration        self.tools = []

        

        # Initialize API client        # Initialize LLM immediately for basic functionality

        self.api_client = APIClient()        try:

        self.meta_agent = None            self.llm = ChatVertexAI(

        self.tool_agent = None                model_name=model,

                temperature=temperature,

        if self.verbose:                project=env.GOOGLE_CLOUD_PROJECT,

            logger.info(f"PlanAgent initialized with model={model}, temperature={temperature}")                location=env.GOOGLE_CLOUD_LOCATION

            )

    async def init_async(self):            logger.info(f"✅ LLM initialized successfully for PlanAgent")

        """Initialize async components if needed"""        except Exception as e:

        pass            logger.error(f"❌ Error initializing LLM: {str(e)}")

            self.llm = None

    def init_sub_agents(self):

        """Initialize MetaAgent and ToolAgent when needed"""        logger.info(f"PlanAgent initialized with model={model}, temperature={temperature}")

        if self.meta_agent is None:

            self.meta_agent = MetaAgent(verbose=self.verbose)        self.max_iteration = max_iteration

        self.verbose = verbose

        if self.tool_agent is None:        self.api_client = APIClient()  # Initialize API client

            self.tool_agent = ToolAgent(verbose=self.verbose)        self.meta_agent = None  # Will be initialized when needed

        self.tool_agent = None  # Will be initialized when needed

    def priority_plan(self, user_input: str) -> Dict[str, Any]:        

        """Create priority-based plans using AutoGen"""        # Initialize graph

        # Get available tools info (placeholder for MCP tools)        self.graph = self.create_graph()

        available_tools = []

        tools_info = "\n".join(available_tools) if available_tools else "- No MCP tools currently available"

    async def init_async(self):

        # Use the structured prompt with tools information        self.tools = await self.get_mcp_tools()

        system_prompt = PLAN_PROMPTS.replace("<<Tools_info>>", tools_info)        self.llm = ChatVertexAI(

            model_name=self.model,

        if self.verbose:            temperature=self.temperature,

            print("Calling LLM to generate priority plans...")            model_kwargs={

                "tools": self.tools

        try:            },

            # Use AutoGen to generate plans            project=env.GOOGLE_CLOUD_PROJECT,

            user_proxy = UserProxyAgent(            location=env.GOOGLE_CLOUD_LOCATION

                name="user_proxy",        )

                code_execution_config=False,    

                human_input_mode="NEVER"    async def get_mcp_tools(self):

            )        async with MultiServerMCPClient({

            "mcp-server": {

            planning_agent = AssistantAgent(                # make sure you start your weather server on port 8000

                name="Planning Agent",                "url": env.MCP_SERVER_URL,

                system_message=system_prompt,                "transport": "sse",

                llm_config=create_vertex_llm_config(self.model, self.temperature),            }

            )        }) as client:

            tools = list(client.get_tools())

            chat_result = user_proxy.initiate_chat(            return tools

                planning_agent,    

                message=f"User request: {user_input}",    def init_sub_agents(self):

                max_turns=1        """Initialize MetaAgent and ToolAgent when needed"""

            )        if self.meta_agent is None:

            self.meta_agent = MetaAgent(

            # Extract response                model=self.model,

            if chat_result and chat_result.chat_history:                temperature=self.temperature,

                response_content = chat_result.chat_history[-1].get('content', '')                verbose=self.verbose

            )

                # Extract plans from response        

                plan_data = extract_priority_plans(response_content)        if self.tool_agent is None:

            self.tool_agent = ToolAgent(

                # Validate that we got plans                model=self.model,

                if not any(plan_data.get(key) for key in ['Security_Plan', 'Convenience_Plan', 'Energy_Plan']):                temperature=self.temperature,

                    if self.verbose:                verbose=self.verbose

                        logger.warning("Warning: No plans extracted from LLM response, using fallback data")            )

                    # Fallback to dummy data if extraction failed            # Initialize ToolAgent async components

                    plan_data = {            try:

                        'Security_Plan': [                import asyncio

                            'Set up security cameras in key areas',                import threading

                            'Install motion sensors and alarm system',                

                            'Enable automated security lighting',                def run_async_init():

                            'Configure door and window sensors'                    # Create new event loop in separate thread

                        ],                    loop = asyncio.new_event_loop()

                        'Convenience_Plan': [                    asyncio.set_event_loop(loop)

                            'Automate lighting based on presence',                    try:

                            'Set up voice control for common tasks',                        loop.run_until_complete(self.tool_agent.init_async())

                            'Create morning and evening routines',                    finally:

                            'Install smart switches for easy control'                        loop.close()

                        ],                

                        'Energy_Plan': [                # Run in separate thread to avoid event loop conflict

                            'Install smart thermostat for climate control',                thread = threading.Thread(target=run_async_init)

                            'Use energy-efficient LED bulbs throughout',                thread.start()

                            'Set up automated power management',                thread.join(timeout=10)  # Wait max 10 seconds

                            'Configure energy monitoring and alerts'                

                        ]                if thread.is_alive():

                    }                    logger.warning("⚠️ ToolAgent async init timeout")

                    

            logger.info("LLM call completed successfully.")            except Exception as e:

        except Exception as e:                logger.warning(f"⚠️ Could not init ToolAgent async: {e}")

            logger.error(f"Error calling LLM: {str(e)}")                logger.info("🧪 ToolAgent will use fallback mode")

            if self.verbose:

                logger.info(f"LLM call failed: {str(e)}, using fallback data")    def router(self, state: PlanState):

            # Use fallback data if LLM call fails        # Check if this is a plan selection

            plan_data = {        selected_plan_id = state.get('selected_plan_id')

                'Security_Plan': [        if selected_plan_id:

                    'Set up security cameras in key areas',            return {**state, 'plan_type': 'execute'}

                    'Install motion sensors and alarm system',        

                    'Enable automated security lighting',        # Check if message indicates plan selection

                    'Configure door and window sensors'        input_msg = state.get('input', '').strip().lower()

                ],        if input_msg in ['plan 1', 'plan 2', 'plan 3', '1', '2', '3', 'plan a', 'plan b', 'plan c', 'a', 'b', 'c']:

                'Convenience_Plan': [            # Extract plan number

                    'Automate lighting based on presence',            if 'plan 1' in input_msg or input_msg == '1' or input_msg == 'plan a' or input_msg == 'a':

                    'Set up voice control for common tasks',                selected_plan_id = 1

                    'Create morning and evening routines',            elif 'plan 2' in input_msg or input_msg == '2' or input_msg == 'plan b' or input_msg == 'b':

                    'Install smart switches for easy control'                selected_plan_id = 2

                ],            elif 'plan 3' in input_msg or input_msg == '3' or input_msg == 'plan c' or input_msg == 'c':

                'Energy_Plan': [                selected_plan_id = 3

                    'Install smart thermostat for climate control',            

                    'Use energy-efficient LED bulbs throughout',            return {**state, 'plan_type': 'execute', 'selected_plan_id': selected_plan_id}

                    'Set up automated power management',        

                    'Configure energy monitoring and alerts'        # Normal plan creation

                ]        routes=[

            }            {

                'route': 'priority',

        # Extract the 3 priority plans                'description': 'This route creates 3 alternative plans prioritized by Security, Convenience, and Energy Efficiency using ONLY MCP smart home tools. The user can review all options and select the most suitable plan. All device information and control operations are handled through MCP tools only.'

        security_plan = plan_data.get('Security_Plan', [])            }

        convenience_plan = plan_data.get('Convenience_Plan', [])        ]

        energy_plan = plan_data.get('Energy_Plan', [])        return {**state,'plan_type':'priority'}



        # Store all plans    def priority_plan(self,state:PlanState):

        plan_options = {        from template.agent.plan.utils import extract_priority_plans

            'security_plan': security_plan,        

            'convenience_plan': convenience_plan,        # Get available MCP tools to include in planning

            'energy_plan': energy_plan        available_tools = []

        }        if self.tools:

            available_tools = [f"- {tool.name}: {tool.description}" for tool in self.tools]  # Limit to first 10 tools

        return {'plan_options': plan_options, 'needs_user_selection': True}        

        tools_info = "\n".join(available_tools) if available_tools else "- No MCP tools currently available"

    def execute_selected_plan(self, user_input: str, selected_plan_id: int, plan_options: dict, token: str = None) -> Dict[str, Any]:        

        """Execute the selected plan with full MetaAgent + ToolAgent workflow"""        # Use the structured prompt with MCP tools information

        if not selected_plan_id:        system_prompt = PLAN_PROMPTS.replace("<<Tools_info>>", tools_info)

            return {'output': 'No plan selected'}

        # Convert custom messages to LangChain messages before passing to LLM

        if not plan_options:        messages = convert_messages_list([

            return {'output': 'No plan options available. Please create a new plan first.'}            SystemMessage(system_prompt),

            HumanMessage(f"User request: {state.get('input')}")

        # Select the plan        ])

        if selected_plan_id == 1:        

            selected_plan = plan_options['security_plan']        logger.info(f"Prompt constructed for Priority Planning:{system_prompt[:50]}...")

            plan_type = 'Security Priority Plan'        

        elif selected_plan_id == 2:        if self.verbose:

            selected_plan = plan_options['convenience_plan']            print("Calling LLM to generate priority plans...")

            plan_type = 'Convenience Priority Plan'        

        elif selected_plan_id == 3:        logger.info("Calling LLM to generate priority plans...")

            selected_plan = plan_options['energy_plan']        

            plan_type = 'Energy Efficiency Priority Plan'        try:

        else:            # Actually call the LLM

            return {'output': 'Invalid plan selection'}            llm_response = self.llm.invoke(messages)

            if self.verbose:

        if self.verbose:                logger.info(colored(f"LLM Response received: {len(llm_response.content)} characters", color='cyan'))

            logger.info(f'✅ Selected Plan: {plan_type}')                # print(colored(f"LLM Response: {llm_response.content}", color='cyan'))

            logger.info('📋 Tasks:')            

            for i, task in enumerate(selected_plan, 1):            # Extract plans from LLM response

                logger.info(f'   {i}. {task}')            plan_data = extract_priority_plans(llm_response.content)

            

        # Initialize sub-agents            # Validate that we got plans

        self.init_sub_agents()            if not any(plan_data.get(key) for key in ['Security_Plan', 'Convenience_Plan', 'Energy_Plan']):

                if self.verbose:

        # Execute plan through MetaAgent + ToolAgent workflow                    logger.warning("Warning: No plans extracted from LLM response, using fallback data")

        execution_results = []                # Fallback to dummy data if extraction failed

        completed_tasks = []                plan_data = {

        failed_tasks = []                    'Security_Plan': [

                        'Set up security cameras in key areas', 

        try:                        'Install motion sensors and alarm system',

            for i, task in enumerate(selected_plan, 1):                        'Enable automated security lighting',

                logger.info(f"\n🚀 Executing Task {i}/{len(selected_plan)}: {task}")                        'Configure door and window sensors'

                    ],

                try:                    'Convenience_Plan': [

                    # MetaAgent analyzes the task                        'Automate lighting based on presence', 

                    meta_input = {                        'Set up voice control for common tasks', 

                        "input": task,                        'Create morning and evening routines',

                        "context": f"This is task {i} of {len(selected_plan)} from {plan_type}",                        'Install smart switches for easy control'

                        "previous_results": execution_results[-3:] if execution_results else []                    ],

                    }                    'Energy_Plan': [

                        'Install smart thermostat for climate control', 

                    meta_result = self.meta_agent.invoke(meta_input)                        'Use energy-efficient LED bulbs throughout', 

                    logger.info(f"🧠 MetaAgent analysis: {meta_result.get('output', 'No output')[:100]}...")                        'Set up automated power management',

                        'Configure energy monitoring and alerts'

                    # ToolAgent executes the specific action                    ]

                    tool_result = self.tool_agent.invoke({"input": task, "token": token})                }



                    if tool_result.get('tool_agent_result', False):            logger.info("LLM call completed successfully.")

                        execution_results.append({        except Exception as e:

                            "task_number": i,            logger.error(f"Error calling LLM: {str(e)}")

                            "task": task,            if self.verbose:

                            "meta_analysis": meta_result.get('output', ''),                logger.info(f"LLM call failed: {str(e)}, using fallback data")

                            "tool_execution": tool_result.get('output', ''),            # Use fallback data if LLM call fails

                            "status": "completed"            plan_data = {

                        })                'Security_Plan': [

                        completed_tasks.append(task)                    'Set up security cameras in key areas', 

                        logger.info(f"✅ Task {i} completed successfully")                    'Install motion sensors and alarm system',

                    else:                    'Enable automated security lighting',

                        error_msg = tool_result.get('error', 'Unknown tool execution error')                    'Configure door and window sensors'

                        execution_results.append({                ],

                            "task_number": i,                'Convenience_Plan': [

                            "task": task,                    'Automate lighting based on presence', 

                            "meta_analysis": meta_result.get('output', ''),                    'Set up voice control for common tasks', 

                            "tool_execution": error_msg,                    'Create morning and evening routines',

                            "status": "failed"                    'Install smart switches for easy control'

                        })                ],

                        failed_tasks.append(task)                'Energy_Plan': [

                        logger.error(f"❌ Task {i} failed: {error_msg}")                    'Install smart thermostat for climate control', 

                    'Use energy-efficient LED bulbs throughout', 

                except Exception as e:                    'Set up automated power management',

                    error_msg = f"Execution error: {str(e)}"                    'Configure energy monitoring and alerts'

                    execution_results.append({                ]

                        "task_number": i,            }

                        "task": task,            logger.error(f"Error calling LLM: {str(e)}")

                        "meta_analysis": "Failed to analyze",            if self.verbose:

                        "tool_execution": error_msg,                logger.info(f"LLM call failed: {str(e)}, using fallback data")

                        "status": "failed"            # Use fallback data if LLM call fails

                    })            plan_data = {

                    failed_tasks.append(task)                'Security_Plan': ['Secure all entry points and doors', 'Set up security cameras in key areas', 'Install motion sensors and alarm system'],

                    logger.error(f"❌ Task {i} failed with exception: {str(e)}")                'Convenience_Plan': ['Automate lighting based on presence', 'Set up voice control for common tasks', 'Create morning and evening routines'],

                'Energy_Plan': ['Install smart thermostat for climate control', 'Use energy-efficient LED bulbs throughout', 'Set up solar panels and energy monitoring']

        except Exception as e:            }

            logger.error(f"❌ Critical error during plan execution: {str(e)}")        

        # Extract the 3 priority plans

        # Generate comprehensive output        security_plan = plan_data.get('Security_Plan', [])

        total_tasks = len(selected_plan)        convenience_plan = plan_data.get('Convenience_Plan', [])

        completed_count = len(completed_tasks)        energy_plan = plan_data.get('Energy_Plan', [])

        failed_count = len(failed_tasks)        

        success_rate = (completed_count / total_tasks * 100) if total_tasks > 0 else 0        # For API mode - return plan options instead of asking for user input

        

        output = f"🎯 **{plan_type} Execution Complete**\n\n"        # Store all plans in state for later use

        output += f"📋 **Summary:**\n"        plan_options = {

        output += f"• Total Tasks: {total_tasks}\n"            'security_plan': security_plan,

        output += f"• Completed: {completed_count}\n"            'convenience_plan': convenience_plan,

        output += f"• Failed: {failed_count}\n"            'energy_plan': energy_plan

        output += f"• Success Rate: {success_rate:.1f}%\n\n"        }

        # Return plan options - will be handled by API response

        if completed_tasks:        return {**state, 'plan_options': plan_options, 'needs_user_selection': True}

            output += f"✅ **Completed Tasks:**\n"

            for task in completed_tasks:

                output += f"• {task}\n"    def initialize(self,state:UpdateState):

            output += "\n"        system_prompt=UPDATE_PLAN_PROMPTS

        plan_list = state.get('plan', [])

        if failed_tasks:        current = plan_list[0] if plan_list else ""

            output += f"❌ **Failed Tasks:**\n"        pending = plan_list or []

            for task in failed_tasks:        completed = []

                output += f"• {task}\n"        

            output += "\n"        if self.verbose:

            if pending:

        output += f"📋 **Detailed Results:**\n"                pending_tasks = "\n".join([f"{index+1}. {task}" for index,task in enumerate(pending)])

        for result in execution_results:                print(colored(f'Pending Tasks:\n{pending_tasks}',color='yellow',attrs=['bold']))

            status_icon = "✅" if result["status"] == "completed" else "❌"            if completed:

            output += f"{status_icon} Task {result['task_number']}: {result['task'][:50]}...\n"                completed_tasks = "\n".join([f"{index+1}. {task}" for index,task in enumerate(completed)])

                print(colored(f'Completed Tasks:\n{completed_tasks}',color='blue',attrs=['bold']))

        return {        

            'plan': selected_plan,        # Khởi tạo tất cả tasks với status "pending" trên API

            'output': output,        if self.api_enabled and self.api_client:

            'execution_results': execution_results,            for task in pending:

            'plan_options': plan_options                self.api_client.update_task_status(task, "pending")

        }            

            # Update plan status to "in_progress" 

    def invoke(self, input_str: str, selected_plan_id: int = None, plan_options: dict = None, token: str = None) -> Dict[str, Any]:            self.api_client.update_plan_status("in_progress")

        """Main entry point for PlanAgent"""        

        start_time = time.time()        messages=[SystemMessage(system_prompt)]

        return {**state,'messages':messages,'current':current,'pending':pending,'completed':completed,'output':''}

        if self.verbose:    

            print(f'Entering ' + colored(self.name, 'black', 'on_white'))    def execute_task(self,state:UpdateState):

        plan=state.get('plan')

        # Check if this is a plan selection        current=state.get('current')

        if selected_plan_id and plan_options:        responses=state.get('responses')

            # Execute selected plan        

            result = self.execute_selected_plan(input_str, selected_plan_id, plan_options, token)        # Update task status to "in_progress" trước khi execute

        else:        if self.api_enabled and self.api_client:

            # Create new plans            self.api_client.update_task_status(current, "in_progress")

            result = self.priority_plan(input_str)        

        agent=MetaAgent(llm=self.llm,verbose=self.verbose)

        return result        responses_text = "\n".join([f"{index+1}. {task}" for index,task in enumerate(responses)])

        task_response=agent.invoke(f'Information:\n{responses_text}\nTask:\n{current}')

    def stream(self, input: str):        

        """Placeholder for streaming"""        if self.verbose:

        pass            print(colored(f'Current Task:\n{current}',color='cyan',attrs=['bold']))
            print(colored(f'Task Response:\n{task_response}',color='cyan',attrs=['bold']))
        
        # Update task với execution result
        if self.api_enabled and self.api_client:
            self.api_client.update_task_status(current, "completed", task_response)
        
        # Truncate task_response if too long to prevent payload issues
        max_response_length = 2000
        if len(task_response) > max_response_length:
            task_response_truncated = task_response[:max_response_length] + "... [response truncated for brevity]"
        else:
            task_response_truncated = task_response
            
        user_prompt=f'Plan:\n{plan}\nTask:\n{current}\nTask Response:\n{task_response_truncated}'
        messages=[HumanMessage(user_prompt)]
        return {**state,'messages':messages,'responses':[task_response]}

    def trim_messages(self, messages, max_tokens=4000):
        """Trim messages to prevent payload too large error"""
        if not messages:
            return messages
            
        # Keep only the most recent messages that fit within token limit
        trimmed = []
        total_chars = 0
        
        # Reverse to process from newest to oldest
        for msg in reversed(messages):
            msg_chars = len(str(msg.content))
            if total_chars + msg_chars > max_tokens * 4:  # Rough char to token ratio
                break
            trimmed.insert(0, msg)
            total_chars += msg_chars
            
        # Always keep at least the last message
        if not trimmed and messages:
            last_msg = messages[-1]
            # Truncate content if too long
            if len(str(last_msg.content)) > max_tokens * 4:
                content = str(last_msg.content)[:max_tokens * 4] + "... [truncated]"
                trimmed = [type(last_msg)(content)]
            else:
                trimmed = [last_msg]
                
        return trimmed

    def update_plan(self,state:UpdateState):
        # Trim messages to prevent payload too large
        messages = self.trim_messages(state.get('messages', []))
        # Convert custom messages to LangChain messages before passing to LLM
        converted_messages = convert_messages_list(messages)
        llm_response=self.llm.invoke(converted_messages)
        plan_data=extract_llm_response(llm_response.content)
        plan=plan_data.get('Plan')
        pending=plan_data.get('Pending') or []  # Default to empty list if None
        completed=plan_data.get('Completed') or []  # Default to empty list if None
        
        if self.verbose:
            if pending:
                pending_tasks = "\n".join([f"{index+1}. {task}" for index,task in enumerate(pending)])
                print(colored(f'Pending Tasks:\n{pending_tasks}',color='yellow',attrs=['bold']))
            if completed:
                completed_tasks = "\n".join([f"{index+1}. {task}" for index,task in enumerate(completed)])
                print(colored(f'Completed Tasks:\n{completed_tasks}',color='blue',attrs=['bold']))
        
        # Update task statuses trên API
        if self.api_enabled and self.api_client:
            # Lấy task trước đó để so sánh
            previous_completed = state.get('completed', [])
            previous_pending = state.get('pending', [])
            
            # Tìm tasks vừa được completed
            newly_completed = [task for task in completed if task not in previous_completed]
            for task in newly_completed:
                self.api_client.update_task_status(task, "completed", f"Task '{task}' hoàn thành thành công")
            
            # Update pending tasks status
            for task in pending:
                if task not in previous_pending:  # New pending task
                    self.api_client.update_task_status(task, "pending")
            
            # Update plan status
            if not pending:  # Tất cả tasks đã completed
                self.api_client.update_plan_status("completed")
            else:
                self.api_client.update_plan_status("in_progress")
        
        if pending:
            current=pending[0]
        else:
            current=''
        # Keep the completed list we already processed instead of overwriting with potentially None
        completed_final = plan_data.get('Completed') or []
        return {**state,'plan':plan,'current':current,'pending':pending,'completed':completed_final}
    
    def final(self,state:UpdateState):
        user_prompt='All Tasks completed successfully. Now give the final answer.'
        # Convert custom messages to LangChain messages before passing to LLM
        messages = convert_messages_list(state.get('messages')+[HumanMessage(user_prompt)])
        llm_response=self.llm.invoke(messages)
        plan_data=extract_llm_response(llm_response.content)
        output=plan_data.get('Final Answer')
        
        # Hoàn thành plan trên API
        if self.api_enabled and self.api_client:
            self.api_client.update_plan_status("completed", output)
            
            if self.verbose:
                print("🎉 Plan execution completed! All data sent to API.")
        
        return {**state,'output':output}

    def execute_selected_plan(self, state: PlanState):
        """Execute the selected plan with full MetaAgent + ToolAgent workflow"""
        selected_plan_id = state.get('selected_plan_id')
        plan_options = state.get('plan_options', {})
        
        if not selected_plan_id:
            return {**state, 'output': 'No plan selected'}
        
        if not plan_options:
            return {**state, 'output': 'No plan options available. Please create a new plan first.'}
        
        # Select the plan
        if selected_plan_id == 1:
            selected_plan = plan_options['security_plan']
            plan_type = 'Security Priority Plan'
        elif selected_plan_id == 2:
            selected_plan = plan_options['convenience_plan']
            plan_type = 'Convenience Priority Plan'
        elif selected_plan_id == 3:
            selected_plan = plan_options['energy_plan']
            plan_type = 'Energy Efficiency Priority Plan'
        else:
            return {**state, 'output': 'Invalid plan selection'}
        
        if self.verbose:
            logger.info(f'✅ Selected Plan: {plan_type}')
            logger.info('📋 Tasks:')
            for i, task in enumerate(selected_plan, 1):
                logger.info(f'   {i}. {task}')
        
        # Step 1: Upload plan to API
        if self.api_client:
            plan_data = {
                "input": state.get('input', ''),
                "plan_type": plan_type.lower().replace(' ', '_'),
                "current_plan": selected_plan,
                "status": "created"
            }
            
            try:
                api_result = self.api_client.create_plan(plan_data)
                if api_result:
                    logger.info(f"📤 Plan uploaded to API successfully")
                    # Update plan status to execution started
                    self.api_client.update_plan_status("in_progress")
                else:
                    logger.warning("⚠️ Failed to upload plan to API, continuing with execution")
            except Exception as e:
                logger.error(f"❌ API upload error: {str(e)}, continuing with execution")
        
        # Step 2: Initialize sub-agents
        self.init_sub_agents()
        
        # Step 3: Execute plan through MetaAgent + ToolAgent workflow
        execution_results = []
        completed_tasks = []
        failed_tasks = []
        
        try:
            for i, task in enumerate(selected_plan, 1):
                logger.info(f"\n🚀 Executing Task {i}/{len(selected_plan)}: {task}")
                
                # Update task status to in_progress
                if self.api_client:
                    self.api_client.update_task_status(task, "in_progress")
                
                # Step 3a: MetaAgent analyzes the task
                meta_input = {
                    "input": task,
                    "context": f"This is task {i} of {len(selected_plan)} from {plan_type}",
                    "previous_results": execution_results[-3:] if execution_results else []  # Last 3 results for context
                }
                
                try:
                    meta_result = self.meta_agent.invoke(meta_input)
                    logger.info(f"🧠 MetaAgent analysis: {meta_result.get('output', 'No output')[:100]}...")
                    
                    # Step 3b: ToolAgent executes the specific action
                    # Pass token to ToolAgent for MCP tool authentication
                    token = state.get('token', '')
                    if token:
                        tool_result = self.tool_agent.invoke({"input": task, "token": token})
                    else:
                        tool_result = self.tool_agent.invoke(task)
                    
                    if tool_result.get('tool_agent_result', False):
                        execution_results.append({
                            "task_number": i,
                            "task": task,
                            "meta_analysis": meta_result.get('output', ''),
                            "tool_execution": tool_result.get('output', ''),
                            "status": "completed"
                        })
                        completed_tasks.append(task)
                        
                        # Update task status to completed
                        if self.api_client:
                            self.api_client.update_task_status(
                                task, 
                                "completed", 
                                tool_result.get('output', '')
                            )
                        
                        logger.info(f"✅ Task {i} completed successfully")
                    else:
                        error_msg = tool_result.get('error', 'Unknown tool execution error')
                        execution_results.append({
                            "task_number": i,
                            "task": task,
                            "meta_analysis": meta_result.get('output', ''),
                            "tool_execution": error_msg,
                            "status": "failed"
                        })
                        failed_tasks.append(task)
                        
                        # Update task status to failed
                        if self.api_client:
                            self.api_client.update_task_status(task, "failed", error_msg)
                        
                        logger.error(f"❌ Task {i} failed: {error_msg}")
                        
                except Exception as e:
                    error_msg = f"MetaAgent execution error: {str(e)}"
                    execution_results.append({
                        "task_number": i,
                        "task": task,
                        "meta_analysis": "Failed to analyze",
                        "tool_execution": error_msg,
                        "status": "failed"
                    })
                    failed_tasks.append(task)
                    
                    if self.api_client:
                        self.api_client.update_task_status(task, "failed", error_msg)
                    
                    logger.error(f"❌ Task {i} failed with exception: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ Critical error during plan execution: {str(e)}")
        
        # Step 4: Finalize and report results
        total_tasks = len(selected_plan)
        completed_count = len(completed_tasks)
        failed_count = len(failed_tasks)
        success_rate = (completed_count / total_tasks * 100) if total_tasks > 0 else 0
        
        # Update final plan status
        if self.api_client:
            final_status = "completed" if failed_count == 0 else "completed"
            final_summary = f"Plan execution completed. {completed_count}/{total_tasks} tasks successful ({success_rate:.1f}%)"
            self.api_client.update_plan_status(final_status, final_summary)
        
        # Generate comprehensive output
        output = f"🎯 **{plan_type} Execution Complete**\n\n"
        output += f"📋 **Summary:**\n"
        output += f"• Total Tasks: {total_tasks}\n"
        output += f"• Completed: {completed_count}\n" 
        output += f"• Failed: {failed_count}\n"
        output += f"• Success Rate: {success_rate:.1f}%\n\n"
        
        if completed_tasks:
            output += f"✅ **Completed Tasks:**\n"
            for task in completed_tasks:
                output += f"• {task}\n"
            output += "\n"
        
        if failed_tasks:
            output += f"❌ **Failed Tasks:**\n"
            for task in failed_tasks:
                output += f"• {task}\n"
            output += "\n"
        
        output += f"📋 **Detailed Results:**\n"
        for result in execution_results:
            status_icon = "✅" if result["status"] == "completed" else "❌"
            output += f"{status_icon} Task {result['task_number']}: {result['task'][:50]}...\n"
        
        return {**state, 'plan': selected_plan, 'output': output, 'execution_results': execution_results}

    def plan_controller(self,state:UpdateState):
        if state.get('pending'):
            return 'task'
        else:
            return 'final'

    def route_controller(self,state:PlanState):
        plan_type = state.get('plan_type')
        if plan_type == 'execute':
            return 'execute_selected'
        elif state.get('needs_user_selection', False):
            return 'wait_selection'
        return plan_type

    def create_graph(self):
        graph=StateGraph(PlanState)
        graph.add_node('route',self.router)
        # Priority planning - MCP tools approach
        graph.add_node('priority',self.priority_plan)
        graph.add_node('execute_selected', self.execute_selected_plan)
        graph.add_node('wait_selection', lambda state: {**state, 'output': 'waiting_for_selection'})
        graph.add_node('execute',lambda _:self.update_graph())

        graph.add_edge(START,'route')
        graph.add_conditional_edges('route',self.route_controller)
        # Handle both direct execution and user selection flow
        graph.add_conditional_edges('priority', lambda state: 'wait_selection' if state.get('needs_user_selection') else 'execute')
        graph.add_edge('execute_selected', END)
        graph.add_edge('wait_selection', END)
        graph.add_edge('execute',END)

        return graph.compile(debug=False)
    
    def update_graph(self):
        graph=StateGraph(UpdateState)
        graph.add_node('inital',self.initialize)
        graph.add_node('task',self.execute_task)
        graph.add_node('update',self.update_plan)
        graph.add_node('final',self.final)

        graph.add_edge(START,'inital')
        graph.add_edge('inital','task')
        graph.add_edge('task','update')
        graph.add_conditional_edges('update',self.plan_controller)
        graph.add_edge('final',END)

        return graph.compile(debug=False)

    def invoke(self,input:str, selected_plan_id: int = None, plan_options: dict = None, token: str = None):
        # Lưu thời gian bắt đầu và input để sử dụng cho API
        self.start_time = time.time()
        self.current_input = input
        self.current_token = token  # Store token for use in sub-agents
        
        if self.verbose:
            print(f'Entering '+colored(self.name,'black','on_white'))
        state={
            'input': input,
            'plan_status':'',
            'route':'',
            'plan': [],
            'plan_options': plan_options or {},  # Use cached plan options if provided
            'needs_user_selection': False,
            'selected_plan_id': selected_plan_id,
            'token': token,  # Add token to state
            'output': ''
        }
        agent_response=self.graph.invoke(state)
        return agent_response

    def select_plan_from_options(self, state: PlanState):
        """Handle plan selection when user provides selection"""
        plan_options = state.get('plan_options', {})
        selected_plan_id = state.get('selected_plan_id')
        
        if selected_plan_id == 1:
            selected_plan = plan_options.get('security_plan', [])
            plan_type = 'priority_security'
        elif selected_plan_id == 2:
            selected_plan = plan_options.get('convenience_plan', [])
            plan_type = 'priority_convenience'
        elif selected_plan_id == 3:
            selected_plan = plan_options.get('energy_plan', [])
            plan_type = 'priority_energy'
        else:
            # Default to security plan
            selected_plan = plan_options.get('security_plan', [])
            plan_type = 'priority_security'
        
        if self.verbose:
            selected_plan_text = "\n".join([f"{index+1}. {task}" for index,task in enumerate(selected_plan)])
            print(colored(f'\nSelected Plan:\n{selected_plan_text}',color='green',attrs=['bold']))
        
        return {**state, 'plan': selected_plan, 'needs_user_selection': False}


    def stream(self, input: str):
        pass