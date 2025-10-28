import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # format thời gian
)
logger = logging.getLogger(__name__)

def extract_plan(response):
    extracted_data = {}
    # Check if it's Option 1 (gathering information)
    option1_match = re.search(r'<option>\s*<question>(.*?)</question>\s*<answer>(.*?)</answer>\s*<route>(.*?)</route>\s*</option>', response, re.DOTALL)
    if option1_match:
        extracted_data['Question'] = option1_match.group(1).strip()
        extracted_data['Answer'] = option1_match.group(2).strip()
        extracted_data['Route'] = option1_match.group(3).strip()
        return extracted_data

    # Check if it's Option 2 (providing the final plan)
    option2_match = re.search(r'<option>\s*<plan>(.*?)</plan>\s*<route>(.*?)</route>\s*</option>', response, re.DOTALL)
    if option2_match:
        # Extract each task from the plan
        plan_content = option2_match.group(1).strip()
        tasks = re.findall(r'(\d+)\.\s*(.*)', plan_content)
        extracted_data['Plan'] = [task[1].strip() for task in tasks]
        extracted_data['Route'] = option2_match.group(2).strip()
        return extracted_data

def extract_llm_response(xml_response: str) -> dict:
    # Initialize the result dictionary
    result = {
        'Current Plan': None,
        'Pending': None,
        'Completed': None,
        'Final Answer': None,
        'Route': None
    }
    
    # Define regex patterns to match each part of the response
    current_plan_regex = re.compile(r'<current-plan>\s*(.*?)\s*</current-plan>', re.DOTALL)
    pending_regex = re.compile(r'<pending>\s*(.*?)\s*</pending>', re.DOTALL)
    completed_regex = re.compile(r'<completed>\s*(.*?)\s*</completed>', re.DOTALL)
    final_answer_regex = re.compile(r'<final-answer>\s*(.*?)\s*</final-answer>', re.DOTALL)
    route_regex = re.compile(r'<route>\s*(.*?)\s*</route>', re.DOTALL)

    # Helper function to clean up task lists by removing bullet points, dashes, and extra characters
    def clean_task_list(task_string: str) -> list:
        # Remove the bullet markers like '- [ ]', '- [x]', and any extra dashes or spaces
        cleaned_tasks = re.sub(r'[-–]\s*\[\s*\]|\[\s*x\s*\]|[-–]\s*', '', task_string)
        # Split the cleaned tasks into a list and strip unnecessary spaces
        return [task.strip() for task in cleaned_tasks.split('\n') if task.strip()]

    # Extract Current Plan
    current_plan_match = current_plan_regex.search(xml_response)
    if current_plan_match:
        current_plan_content = current_plan_match.group(1).strip()
        # Remove numeric prefixes like "1.", "2.", etc.
        result['Current Plan'] = [re.sub(r'^\d+\.\s*', '', task).strip() for task in current_plan_content.split('\n') if task.strip()]

    # Extract Pending tasks
    pending_match = pending_regex.search(xml_response)
    if pending_match:
        pending_content = pending_match.group(1).strip()
        result['Pending'] = clean_task_list(pending_content)

    # Extract Completed tasks
    completed_match = completed_regex.search(xml_response)
    if completed_match:
        completed_content = completed_match.group(1).strip()
        result['Completed'] = clean_task_list(completed_content)

    # Extract Final Answer
    final_answer_match = final_answer_regex.search(xml_response)
    if final_answer_match:
        result['Final Answer'] = final_answer_match.group(1).strip()

    # Extract Route
    route_match = route_regex.search(xml_response)
    if route_match:
        result['Route'] = route_match.group(1).strip()

    return result

def extract_priority_plans(response):
    """Extract the 3 priority plans from XML format response"""
    extracted_data = {
        'Security_Plan': [],
        'Convenience_Plan': [],
        'Energy_Plan': []
    }
    
    # First try XML format parsing
    xml_patterns = [
        (r'<Security_Plan>\s*(.*?)\s*</Security_Plan>', 'Security_Plan'),
        (r'<Convenience_Plan>\s*(.*?)\s*</Convenience_Plan>', 'Convenience_Plan'),
        (r'<Energy_Plan>\s*(.*?)\s*</Energy_Plan>', 'Energy_Plan')
    ]
    
    for pattern, plan_key in xml_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            plan_content = match.group(1).strip()
            # Extract tasks - look for lines starting with "-" or numbers
            tasks = []
            for line in plan_content.split('\n'):
                line = line.strip()
                if line:
                    # Remove bullet points, dashes, numbers
                    clean_line = re.sub(r'^[-•\*]\s*', '', line)  # Remove bullet points
                    clean_line = re.sub(r'^\d+\.\s*', '', clean_line)  # Remove numbers
                    if clean_line:
                        tasks.append(clean_line)
            extracted_data[plan_key] = tasks
    
    # If XML parsing didn't work, try fallback methods
    if not any(extracted_data.values()):
        # Split response into lines for easier processing
        lines = response.split('\n')
        current_plan = None
        
        for line in lines:
            line = line.strip()
            
            # Look for plan headers
            if "Plan 1:" in line or "Maximum Security" in line or "🥇" in line or "Security" in line:
                current_plan = 'Security_Plan'
            elif "Plan 2:" in line or "Balanced Comfort" in line or "🥈" in line or "Convenience" in line:
                current_plan = 'Convenience_Plan'
            elif "Plan 3:" in line or "Energy-Efficient" in line or "🥉" in line or "Energy" in line:
                current_plan = 'Energy_Plan'
            # Look for numbered tasks or bullet points
            elif current_plan and (re.match(r'^\d+\.', line) or re.match(r'^[-•\*]', line)):
                # Extract task text (remove number prefix and bullets)
                task = re.sub(r'^\d+\.\s*', '', line)
                task = re.sub(r'^[-•\*]\s*', '', task)
                if task:  # Only add non-empty tasks
                    extracted_data[current_plan].append(task)
        
        # If we still couldn't find plans, try pattern matching
        if not any(extracted_data.values()):
            plan_patterns = [
                (r'(?:plan\s+1|security|🥇).*?(?=plan\s+2|convenience|🥈|$)', 'Security_Plan'),
                (r'(?:plan\s+2|convenience|🥈).*?(?=plan\s+3|energy|🥉|$)', 'Convenience_Plan'),
                (r'(?:plan\s+3|energy|🥉).*?(?=plan\s+4|custom|$)', 'Energy_Plan')
            ]
            
            for pattern, plan_key in plan_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    plan_text = match.group(0)
                    tasks = re.findall(r'(?:^\d+\.\s*|^[-•\*]\s*)(.*)', plan_text, re.MULTILINE)
                    extracted_data[plan_key] = [task.strip() for task in tasks if task.strip()]
    
    logger.info(f"Extracted Plans: {extracted_data}")
    return extracted_data

def read_markdown_file(file_path: str) -> str:
    with open(file_path, 'r',encoding='utf-8') as f:
        markdown_content = f.read()
    return markdown_content