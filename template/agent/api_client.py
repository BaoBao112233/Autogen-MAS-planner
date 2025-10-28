import requests
import json
import uuid
from typing import List, Dict, Optional
from os import environ
from dotenv import load_dotenv
from datetime import datetime
from template.configs.environments import env
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

class APIClient:
    """Client để gửi thông tin plan và task status lên Planner API server"""
    
    def __init__(self):
        self.base_url = env.PLAN_API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.current_plan_id = None
        self.current_task_ids = {}  # mapping task title -> task_id
        self.session_id = 1

        self.headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7',
            'content-type': 'application/json',
            'origin': 'https://ow-subscription-api.smarthiz.com',
            'priority': 'u=1, i',
            'referer': 'https://ow-subscription-api.smarthiz.com/console',
            'sec-ch-ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            'x-hasura-admin-secret': env.PLAN_API_KEY
        }

    def create_plan(self, plan_data: Dict) -> Optional[Dict]:
        """
        Tạo plan mới với các tasks
        
        Args:
            plan_data: Dictionary chứa thông tin plan
            {
                "input": str,
                "plan_type": str,
                "current_plan": List[str],
                "status": str
            }
        """
        try:

            mutation = """
            mutation createPlan($objects: [planner_plans_insert_input!]!) {
                insert_planner_plans(objects: $objects) {
                    affected_rows
                    returning {
                        id
                        title
                        goal_text
                        trigger
                        priority
                        tasks {
                            id
                            order_no
                            title
                            description
                            max_retries
                        }
                    }
                }
            }
            """

            # Chuyển đổi current_plan thành tasks format
            tasks = []
            for index, task_title in enumerate(plan_data.get("current_plan", [])):
                tasks.append({
                    "order_no": index + 1,
                    "title": task_title,
                    "description": f"Task từ {plan_data.get('plan_type', 'unknown')} plan",
                    "max_retries": 2
                })    
            
            title_plan = f"Plan Agent - {plan_data.get('plan_type', 'Unknown').title()}"
            # Tạo plan payload theo format API
            plan_payload = [{
                "session_id": self.session_id,
                "title": title_plan,
                "goal_text": plan_data.get("input", ""),
                "trigger": "SYSTEM",
                "priority": 1,
                "tasks": {
                    "data": tasks
                }
            }]

            variables = {
                "objects": plan_payload
            }
                
            payload = {
                "query": mutation,
                "variables": variables,
                "operationName": "createPlan"
            }

            headers = self.headers
            
            response = requests.post(env.PLAN_API_BASE_URL, headers=headers, json=payload)
        
            if response.status_code == 200:
                data = response.json()
                if 'errors' in data:
                    logger.error(f"Task '{title_plan}' creation failed: {data['errors']}")
                    return None
                else:
                    # Extract plan and task IDs from response
                    plan_info = data['data']['insert_planner_plans']['returning'][0]
                    plan_id = plan_info['id']
                    
                    # Store current plan ID
                    self.current_plan_id = plan_id
                    logger.info(f"Task '{title_plan}' created successfully with ID: {plan_id}")
                    
                    # Store task IDs mapping
                    self.current_task_ids = {}
                    if 'tasks' in plan_info and plan_info['tasks']:
                        for task in plan_info['tasks']:
                            task_title = task['title']
                            task_id = task['id']
                            self.current_task_ids[task_title] = task_id
                            logger.info(f"📋 Task '{task_title}' mapped to ID: {task_id}")
                    
                    logger.info(f"📊 Plan tracking initialized: Plan ID={plan_id}, Task IDs={len(self.current_task_ids)}")
                    return data
            else:
                logger.error(f"Task '{title_plan}' creation failed: HTTP error: {response.status_code} - {response.text}")
                return None
            
        except Exception as e:
            logger.error(f"Plan creation failed for '{title_plan}': {str(e)}")
            return None

    def update_plan_status(self, status: str, goal_text: str = None) -> Optional[Dict]:
        """
        Cập nhật status của plan
        
        Args:
            status: "created", "in_progress", "completed", "failed"
            goal_text: Updated goal text (optional)
        """
        if not self.current_plan_id:
            print("❌ No current plan ID to update")
            return None
            
        try:
            if status == "in_progress":
                status = "RUNNING"
            if status == "completed":
                status = "DONE"
            if status == "failed":
                status = "FAILED"
            if status == "created":
                status = "PENDING"

            logger.info(f"Updating plan {self.current_plan_id} to status: {status}")

            mutation = f"""
            mutation UpdateStatusPlans {{
                update_planner_plans_by_pk(pk_columns: {{ id: \"{self.current_plan_id}\" }}, _set: {{ status: \"{status}\" }}) {{
                    id
                    title
                    goal_text
                    trigger
                    priority
                    status
                    tasks {{
                        id
                        order_no
                        title
                        description
                        max_retries
                        status
                    }}
                }}
            }}
            """

            payload = {
                "query": mutation,
                "variables": None,
                "operationName": "UpdateStatusPlans"
            }

            headers = self.headers

            response = requests.post(env.PLAN_API_BASE_URL, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()

                if 'errors' in data:
                    logger.error(f"Plan '{self.current_plan_id}' update failed: {data['errors']}")
                    return None
                
                else:
                    logger.info(f"Plan '{self.current_plan_id}' updated successfully to status: {status}")
                    return data
            
            else:
                logger.error(f"Plan '{self.current_plan_id}' update failed: HTTP error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Plan update failed for '{self.current_plan_id}': {str(e)}")
            return None

    def update_task_status(self, task_title: str, status: str, execution_result: str = None) -> Optional[Dict]:
        """
        Cập nhật status của task
        
        Args:
            task_title: Tên của task
            status: "pending", "in_progress", "completed", "failed"
            execution_result: Kết quả thực thi task
        """
        # Direct lookup first
        task_id = self.current_task_ids.get(task_title)
        
        # If not found, try fuzzy matching
        if not task_id:
            # Try to find by partial match or similarity
            for stored_title, stored_id in self.current_task_ids.items():
                # Check if one is a substring of the other (case insensitive)
                if (task_title.lower() in stored_title.lower() or 
                    stored_title.lower() in task_title.lower()):
                    task_id = stored_id
                    logger.info(f"🔍 Found task by partial match: '{task_title}' -> '{stored_title}' (ID: {task_id})")
                    break
        
        if not task_id:
            print(f"❌ No task ID found for task: {task_title}")
            print(f"📋 Available tasks: {list(self.current_task_ids.keys())}")
            return None
            
        try:

            if status == "in_progress":
                status = "RUNNING"
            if status == "completed":
                status = "DONE"
            if status == "failed":
                status = "FAILED"
            if status == "pending":
                status = "DRAFT"

            query = f"""
            mutation UpdateStatusTasks {{
                update_planner_tasks_by_pk(pk_columns: {{id: \"{task_id}\"}}, _set: {{status: \"{status}\"}}) {{
                    id
                    order_no
                    title
                    description
                    max_retries
                    status
                }}
            }}
            """

            payload = {
                "query": query,
                "variables": None,
                "operationName": "UpdateStatusTasks"
            }

            headers = self.headers

            response = requests.post(env.PLAN_API_BASE_URL, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                if 'errors' in data:
                    logger.error(f"Task '{task_id}' update failed: {data['errors']}")
                    return None
                else:
                    logger.info(f"Task '{task_id}' updated successfully to status: {status}")
                    return data
            else:
                logger.error(f"Task '{task_id}' update failed: HTTP error: {response.status_code} - {response.text}")
                return None
            
        except Exception as e:
            logger.error(f"Task update failed for '{task_id}': {str(e)}")
            return None

    def get_plan(self, plan_id: str = None) -> Optional[Dict]:
        """Lấy thông tin plan"""
        plan_id = plan_id or self.current_plan_id
        try:
            logger.info(f"Retrieving plan with ID: {plan_id}")

            query = f"query GetPlanById {{ planner_plans_by_pk(id: \"{plan_id}\") {{ id title goal_text trigger priority status tasks {{ id order_no title description max_retries status }} }} }}"
            payload = {
                "query": query,
                "variables": None,
                "operationName": "GetPlanById"
            }

            headers = self.headers

            response = requests.post(env.PLAN_API_BASE_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                if 'errors' in data:
                    logger.error(f"Plan '{plan_id}' retrieval failed: {data['errors']}")
                    return None
                else:
                    logger.info(f"Plan '{plan_id}' retrieved successfully")
                    return data
            else:
                logger.error(f"Plan '{plan_id}' retrieval failed: HTTP error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Plan retrieval failed for '{plan_id}': {str(e)}")
            return None

    # Backward compatibility methods
    def send_plan_status(self, plan_data: Dict) -> Optional[Dict]:
        """Compatibility method - tạo plan hoặc update status"""
        status = plan_data.get("status", "created")
        
        if status == "plan_created":
            return self.create_plan(plan_data)
        elif status == "execution_started":
            return self.update_plan_status("in_progress")
        elif status == "plan_updated":
            return self.update_plan_status("in_progress")
        else:
            return self.update_plan_status(status)
    
    def send_task_update(self, task_data: Dict) -> Optional[Dict]:
        """Compatibility method - update task"""
        task_name = task_data.get("task_name", "")
        task_status = "completed" if task_data.get("status") == "task_completed" else "in_progress"
        execution_result = task_data.get("task_response", "")
        
        return self.update_task_status(task_name, task_status, execution_result)
    
    def send_final_result(self, result_data: Dict) -> Optional[Dict]:
        """Compatibility method - finalize plan"""
        return self.update_plan_status("completed", result_data.get("final_answer"))