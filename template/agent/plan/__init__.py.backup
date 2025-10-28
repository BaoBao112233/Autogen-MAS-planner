from template.agent import BaseAgent
from template.agent.plan.state import PlanState,UpdateState
# from template.agent.meta.agent import MetaAgent
from template.message.message import SystemMessage,HumanMessage
from template.message.converter import convert_messages_list
from template.agent.plan.utils import (
    extract_llm_response, 
    read_markdown_file,
    extract_priority_plans
)
from template.agent.plan.prompts import PLAN_PROMPTS, UPDATE_PLAN_PROMPTS
from template.agent.meta import MetaAgent
from template.agent.tool import ToolAgent
from template.agent.api_client import APIClient
from template.configs.environments import env

from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph,END,START
from langchain_mcp_adapters.client import MultiServerMCPClient
from termcolor import colored
import os
import time
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # format thời gian
)
logger = logging.getLogger(__name__)

class PlanAgent(BaseAgent):
    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.2, max_iteration=10,verbose=True):
        super().__init__()
        
        self.name = "Plan Agent"

        self.model = model
        self.temperature = temperature
        self.verbose = verbose

        self.tools = []
        
        # Initialize LLM immediately for basic functionality
        try:
            self.llm = ChatVertexAI(
                model_name=model,
                temperature=temperature,
                project=env.GOOGLE_CLOUD_PROJECT,
                location=env.GOOGLE_CLOUD_LOCATION
            )
            logger.info(f"✅ LLM initialized successfully for PlanAgent")
        except Exception as e:
            logger.error(f"❌ Error initializing LLM: {str(e)}")
            self.llm = None

        logger.info(f"PlanAgent initialized with model={model}, temperature={temperature}")

        self.max_iteration = max_iteration
        self.verbose = verbose
        self.api_client = APIClient()  # Initialize API client
        self.meta_agent = None  # Will be initialized when needed
        self.tool_agent = None  # Will be initialized when needed
        
        # Initialize graph
        self.graph = self.create_graph()


    async def init_async(self):
        self.tools = await self.get_mcp_tools()
        self.llm = ChatVertexAI(
            model_name=self.model,
            temperature=self.temperature,
            model_kwargs={
                "tools": self.tools
            },
            project=env.GOOGLE_CLOUD_PROJECT,
            location=env.GOOGLE_CLOUD_LOCATION
        )
    
    async def get_mcp_tools(self):
        async with MultiServerMCPClient({
            "mcp-server": {
                # make sure you start your weather server on port 8000
                "url": env.MCP_SERVER_URL,
                "transport": "sse",
            }
        }) as client:
            tools = list(client.get_tools())
            return tools
    
    def init_sub_agents(self):
        """Initialize MetaAgent and ToolAgent when needed"""
        if self.meta_agent is None:
            self.meta_agent = MetaAgent(
                model=self.model,
                temperature=self.temperature,
                verbose=self.verbose
            )
        
        if self.tool_agent is None:
            self.tool_agent = ToolAgent(
                model=self.model,
                temperature=self.temperature,
                verbose=self.verbose
            )
            # Initialize ToolAgent async components
            try:
                import asyncio
                import threading
                
                def run_async_init():
                    # Create new event loop in separate thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.tool_agent.init_async())
                    finally:
                        loop.close()
                
                # Run in separate thread to avoid event loop conflict
                thread = threading.Thread(target=run_async_init)
                thread.start()
                thread.join(timeout=10)  # Wait max 10 seconds
                
                if thread.is_alive():
                    logger.warning("⚠️ ToolAgent async init timeout")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not init ToolAgent async: {e}")
                logger.info("🧪 ToolAgent will use fallback mode")

    def router(self, state: PlanState):
        # Check if this is a plan selection
        selected_plan_id = state.get('selected_plan_id')
        if selected_plan_id:
            return {**state, 'plan_type': 'execute'}
        
        # Check if message indicates plan selection
        input_msg = state.get('input', '').strip().lower()
        if input_msg in ['plan 1', 'plan 2', 'plan 3', '1', '2', '3', 'plan a', 'plan b', 'plan c', 'a', 'b', 'c']:
            # Extract plan number
            if 'plan 1' in input_msg or input_msg == '1' or input_msg == 'plan a' or input_msg == 'a':
                selected_plan_id = 1
            elif 'plan 2' in input_msg or input_msg == '2' or input_msg == 'plan b' or input_msg == 'b':
                selected_plan_id = 2
            elif 'plan 3' in input_msg or input_msg == '3' or input_msg == 'plan c' or input_msg == 'c':
                selected_plan_id = 3
            
            return {**state, 'plan_type': 'execute', 'selected_plan_id': selected_plan_id}
        
        # Normal plan creation
        routes=[
            {
                'route': 'priority',
                'description': 'This route creates 3 alternative plans prioritized by Security, Convenience, and Energy Efficiency using ONLY MCP smart home tools. The user can review all options and select the most suitable plan. All device information and control operations are handled through MCP tools only.'
            }
        ]
        return {**state,'plan_type':'priority'}

    def priority_plan(self,state:PlanState):
        from template.agent.plan.utils import extract_priority_plans
        
        # Get available MCP tools to include in planning
        available_tools = []
        if self.tools:
            available_tools = [f"- {tool.name}: {tool.description}" for tool in self.tools]  # Limit to first 10 tools
        
        tools_info = "\n".join(available_tools) if available_tools else "- No MCP tools currently available"
        
        # Use the structured prompt with MCP tools information
        system_prompt = PLAN_PROMPTS.replace("<<Tools_info>>", tools_info)

        # Convert custom messages to LangChain messages before passing to LLM
        messages = convert_messages_list([
            SystemMessage(system_prompt),
            HumanMessage(f"User request: {state.get('input')}")
        ])
        
        logger.info(f"Prompt constructed for Priority Planning:{system_prompt[:50]}...")
        
        if self.verbose:
            print("Calling LLM to generate priority plans...")
        
        logger.info("Calling LLM to generate priority plans...")
        
        try:
            # Actually call the LLM
            llm_response = self.llm.invoke(messages)
            if self.verbose:
                logger.info(colored(f"LLM Response received: {len(llm_response.content)} characters", color='cyan'))
                # print(colored(f"LLM Response: {llm_response.content}", color='cyan'))
            
            # Extract plans from LLM response
            plan_data = extract_priority_plans(llm_response.content)
            
            # Validate that we got plans
            if not any(plan_data.get(key) for key in ['Security_Plan', 'Convenience_Plan', 'Energy_Plan']):
                if self.verbose:
                    logger.warning("Warning: No plans extracted from LLM response, using fallback data")
                # Fallback to dummy data if extraction failed
                plan_data = {
                    'Security_Plan': [
                        'Set up security cameras in key areas', 
                        'Install motion sensors and alarm system',
                        'Enable automated security lighting',
                        'Configure door and window sensors'
                    ],
                    'Convenience_Plan': [
                        'Automate lighting based on presence', 
                        'Set up voice control for common tasks', 
                        'Create morning and evening routines',
                        'Install smart switches for easy control'
                    ],
                    'Energy_Plan': [
                        'Install smart thermostat for climate control', 
                        'Use energy-efficient LED bulbs throughout', 
                        'Set up automated power management',
                        'Configure energy monitoring and alerts'
                    ]
                }

            logger.info("LLM call completed successfully.")
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            if self.verbose:
                logger.info(f"LLM call failed: {str(e)}, using fallback data")
            # Use fallback data if LLM call fails
            plan_data = {
                'Security_Plan': [
                    'Set up security cameras in key areas', 
                    'Install motion sensors and alarm system',
                    'Enable automated security lighting',
                    'Configure door and window sensors'
                ],
                'Convenience_Plan': [
                    'Automate lighting based on presence', 
                    'Set up voice control for common tasks', 
                    'Create morning and evening routines',
                    'Install smart switches for easy control'
                ],
                'Energy_Plan': [
                    'Install smart thermostat for climate control', 
                    'Use energy-efficient LED bulbs throughout', 
                    'Set up automated power management',
                    'Configure energy monitoring and alerts'
                ]
            }
            logger.error(f"Error calling LLM: {str(e)}")
            if self.verbose:
                logger.info(f"LLM call failed: {str(e)}, using fallback data")
            # Use fallback data if LLM call fails
            plan_data = {
                'Security_Plan': ['Secure all entry points and doors', 'Set up security cameras in key areas', 'Install motion sensors and alarm system'],
                'Convenience_Plan': ['Automate lighting based on presence', 'Set up voice control for common tasks', 'Create morning and evening routines'],
                'Energy_Plan': ['Install smart thermostat for climate control', 'Use energy-efficient LED bulbs throughout', 'Set up solar panels and energy monitoring']
            }
        
        # Extract the 3 priority plans
        security_plan = plan_data.get('Security_Plan', [])
        convenience_plan = plan_data.get('Convenience_Plan', [])
        energy_plan = plan_data.get('Energy_Plan', [])
        
        # For API mode - return plan options instead of asking for user input
        
        # Store all plans in state for later use
        plan_options = {
            'security_plan': security_plan,
            'convenience_plan': convenience_plan,
            'energy_plan': energy_plan
        }
        # Return plan options - will be handled by API response
        return {**state, 'plan_options': plan_options, 'needs_user_selection': True}


    def initialize(self,state:UpdateState):
        system_prompt=UPDATE_PLAN_PROMPTS
        plan_list = state.get('plan', [])
        current = plan_list[0] if plan_list else ""
        pending = plan_list or []
        completed = []
        
        if self.verbose:
            if pending:
                pending_tasks = "\n".join([f"{index+1}. {task}" for index,task in enumerate(pending)])
                print(colored(f'Pending Tasks:\n{pending_tasks}',color='yellow',attrs=['bold']))
            if completed:
                completed_tasks = "\n".join([f"{index+1}. {task}" for index,task in enumerate(completed)])
                print(colored(f'Completed Tasks:\n{completed_tasks}',color='blue',attrs=['bold']))
        
        # Khởi tạo tất cả tasks với status "pending" trên API
        if self.api_enabled and self.api_client:
            for task in pending:
                self.api_client.update_task_status(task, "pending")
            
            # Update plan status to "in_progress" 
            self.api_client.update_plan_status("in_progress")
        
        messages=[SystemMessage(system_prompt)]
        return {**state,'messages':messages,'current':current,'pending':pending,'completed':completed,'output':''}
    
    def execute_task(self,state:UpdateState):
        plan=state.get('plan')
        current=state.get('current')
        responses=state.get('responses')
        
        # Update task status to "in_progress" trước khi execute
        if self.api_enabled and self.api_client:
            self.api_client.update_task_status(current, "in_progress")
        
        agent=MetaAgent(llm=self.llm,verbose=self.verbose)
        responses_text = "\n".join([f"{index+1}. {task}" for index,task in enumerate(responses)])
        task_response=agent.invoke(f'Information:\n{responses_text}\nTask:\n{current}')
        
        if self.verbose:
            print(colored(f'Current Task:\n{current}',color='cyan',attrs=['bold']))
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