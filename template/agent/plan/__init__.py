"""
Plan Agent for MAS-Planning system using AutoGen

The Plan Agent creates smart home automation plans using AutoGen agents
instead of LangChain/LangGraph workflows.
"""

from template.agent import BaseAgent
from template.agent.autogen_config import create_autogen_agent, create_user_proxy_agent
from template.agent.plan.prompts import PLAN_PROMPTS, UPDATE_PLAN_PROMPTS
from template.agent.plan.utils import (
    read_markdown_file,
    extract_llm_response,
    extract_priority_plans
)
from template.agent.meta import MetaAgent
from template.agent.tool import ToolAgent
from template.agent.api_client import APIClient
from template.configs.environments import env

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from termcolor import colored
import os
import time
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class PlanAgent(BaseAgent):
    """
    Plan Agent - Creates smart home automation plans using AutoGen

    Uses AutoGen GroupChat for coordinating planning activities instead of
    LangGraph StateGraph workflows.
    """

    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.2, max_iteration=10, verbose=True):
        super().__init__()

        self.name = "Plan Agent"
        self.model = model
        self.temperature = temperature
        self.max_iteration = max_iteration
        self.verbose = verbose

        # Initialize API client
        self.api_client = APIClient()

        # Lazy-loaded sub-agents
        self._meta_agent = None
        self._tool_agent = None

        # Create AutoGen planning agent
        self.planning_agent = create_autogen_agent(
            name="Planning Agent",
            system_message=PLAN_PROMPTS,
            model=model,
            temperature=temperature
        )

        # Create user proxy for planning interactions
        self.user_proxy = create_user_proxy_agent(
            name="planning_user_proxy",
            human_input_mode="NEVER"
        )

        if self.verbose:
            logger.info(f"✅ PlanAgent initialized with model={model}, temperature={temperature}")

    async def init_async(self):
        """Initialize async components if needed"""
        pass

    @property
    def meta_agent(self):
        """Lazy load MetaAgent"""
        if self._meta_agent is None:
            self._meta_agent = MetaAgent(verbose=self.verbose)
        return self._meta_agent

    @property
    def tool_agent(self):
        """Lazy load ToolAgent"""
        if self._tool_agent is None:
            self._tool_agent = ToolAgent(verbose=self.verbose)
        return self._tool_agent

    def priority_plan(self, user_input: str) -> dict:
        """
        Create priority-based plans using AutoGen
        """
        if self.verbose:
            logger.info("Calling AutoGen to generate priority plans...")

        try:
            # Get available tools info (placeholder for MCP tools)
            available_tools = []
            tools_info = "\n".join(available_tools) if available_tools else "- No MCP tools currently available"

            # Use the structured prompt with tools information
            system_prompt = PLAN_PROMPTS.replace("<<Tools_info>>", tools_info)

            # Update planning agent with current tools info
            self.planning_agent.update_system_message(system_prompt)

            # Initiate chat with planning agent
            chat_result = self.user_proxy.initiate_chat(
                self.planning_agent,
                message=f"User request: {user_input}",
                max_turns=3
            )

            # Extract response
            if chat_result and chat_result.chat_history:
                response_content = chat_result.chat_history[-1].get('content', '')

                # Extract plans from response
                plan_data = extract_priority_plans(response_content)

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

                logger.info("AutoGen planning call completed successfully.")

                # Extract the 3 priority plans
                security_plan = plan_data.get('Security_Plan', [])
                convenience_plan = plan_data.get('Convenience_Plan', [])
                energy_plan = plan_data.get('Energy_Plan', [])

                # Store all plans
                plan_options = {
                    'security_plan': security_plan,
                    'convenience_plan': convenience_plan,
                    'energy_plan': energy_plan
                }

                return {'plan_options': plan_options, 'needs_user_selection': True}

        except Exception as e:
            logger.error(f"Error calling AutoGen planning: {str(e)}")

            if self.verbose:
                logger.info(f"AutoGen planning failed: {str(e)}, using fallback data")

            # Use fallback data if AutoGen call fails
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

            # Extract the 3 priority plans
            security_plan = plan_data.get('Security_Plan', [])
            convenience_plan = plan_data.get('Convenience_Plan', [])
            energy_plan = plan_data.get('Energy_Plan', [])

            # Store all plans
            plan_options = {
                'security_plan': security_plan,
                'convenience_plan': convenience_plan,
                'energy_plan': energy_plan
            }

            return {'plan_options': plan_options, 'needs_user_selection': True}

    def execute_selected_plan(self, user_input: str, selected_plan_id: int, plan_options: dict, token: str = None) -> dict:
        """
        Execute the selected plan with full MetaAgent + ToolAgent workflow
        """
        if not selected_plan_id:
            return {'output': 'No plan selected'}

        if not plan_options:
            return {'output': 'No plan options available. Please create a new plan first.'}

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
            return {'output': 'Invalid plan selection'}

        if self.verbose:
            logger.info(f'✅ Selected Plan: {plan_type}')
            logger.info('📋 Tasks:')
            for i, task in enumerate(selected_plan, 1):
                logger.info(f'   {i}. {task}')

        # Upload plan to API if available
        if self.api_client:
            plan_data = {
                "input": user_input,
                "plan_type": plan_type.lower().replace(' ', '_'),
                "current_plan": selected_plan,
                "status": "created"
            }

            try:
                api_result = self.api_client.create_plan(plan_data)
                if api_result:
                    logger.info(f"📤 Plan uploaded to API successfully")
                    self.api_client.update_plan_status("in_progress")
                else:
                    logger.warning("⚠️ Failed to upload plan to API, continuing with execution")
            except Exception as e:
                logger.error(f"❌ API upload error: {str(e)}, continuing with execution")

        # Execute plan through MetaAgent + ToolAgent workflow
        execution_results = []
        completed_tasks = []
        failed_tasks = []

        try:
            for i, task in enumerate(selected_plan, 1):
                logger.info(f"\n🚀 Executing Task {i}/{len(selected_plan)}: {task}")

                # Update task status to in_progress
                if self.api_client:
                    self.api_client.update_task_status(task, "in_progress")

                try:
                    # MetaAgent analyzes the task
                    meta_input = {
                        "input": task,
                        "context": f"This is task {i} of {len(selected_plan)} from {plan_type}",
                        "previous_results": execution_results[-3:] if execution_results else []
                    }

                    meta_result = self.meta_agent.invoke(meta_input)
                    logger.info(f"🧠 MetaAgent analysis: {meta_result.get('output', 'No output')[:100]}...")

                    # ToolAgent executes the specific action
                    tool_input = {"input": task, "token": token} if token else task
                    tool_result = self.tool_agent.invoke(tool_input)

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
                    error_msg = f"Execution error: {str(e)}"
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

        # Finalize and report results
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

        return {
            'plan': selected_plan,
            'output': output,
            'execution_results': execution_results,
            'plan_options': plan_options
        }

    def invoke(self, input_str: str, selected_plan_id: int = None, plan_options: dict = None, token: str = None) -> dict:
        """
        Main entry point for PlanAgent
        """
        start_time = time.time()

        if self.verbose:
            print(f'Entering ' + colored(self.name, 'black', 'on_white'))

        if selected_plan_id and plan_options:
            # Execute selected plan
            result = self.execute_selected_plan(input_str, selected_plan_id, plan_options, token)
        else:
            # Create new plans
            result = self.priority_plan(input_str)

        return result

    def stream(self, input: str):
        """Placeholder for streaming"""
        pass
