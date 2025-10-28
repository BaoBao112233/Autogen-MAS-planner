"""
Utilities for logging and timing
"""
import logging
import time
import asyncio
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log execution time of a function"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        func_name = func.__name__
        logger.info(f"🚀 Starting {func_name}")
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"✅ Completed {func_name} in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Failed {func_name} in {execution_time:.2f}s: {str(e)}")
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        func_name = func.__name__
        logger.info(f"🚀 Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"✅ Completed {func_name} in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Failed {func_name} in {execution_time:.2f}s: {str(e)}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def log_with_traceback(func: Callable) -> Callable:
    """Decorator to log exceptions with full traceback"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            import traceback
            logger.error(f"❌ Exception in {func.__name__}: {str(e)}")
            logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            logger.error(f"❌ Exception in {func.__name__}: {str(e)}")
            logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def setup_detailed_logging():
    """Setup detailed logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug.log', mode='a')
        ]
    )