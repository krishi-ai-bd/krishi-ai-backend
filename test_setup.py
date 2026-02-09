"""
Test script to verify Krishi AI backend setup
Run this after installation to check if everything is configured correctly
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("🔍 Testing package imports...")
    
    packages = [
        ('fastapi', 'FastAPI'),
        ('openai', 'OpenAI'),
        ('redis', 'Redis'),
        ('chromadb', 'ChromaDB'),
        ('tiktoken', 'Tiktoken'),
        ('PyPDF2', 'PyPDF2'),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed correctly\n")
    return True


def test_env_config():
    """Test if environment variables are configured"""
    print("🔍 Testing environment configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key'
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value == "your_openai_api_key_here":
            print(f"  ✗ {description} - NOT SET")
            missing.append(var)
        else:
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  ✓ {description}: {masked_value}")
    
    # Optional vars
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    print(f"  ℹ Redis URL: {redis_url}")
    
    chroma_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
    print(f"  ℹ ChromaDB Directory: {chroma_dir}")
    
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("Please configure them in your .env file")
        return False
    
    print("\n✅ Environment configured correctly\n")
    return True


def test_redis_connection():
    """Test Redis connection"""
    print("🔍 Testing Redis connection...")
    
    try:
        import redis
        from app.core.config import settings
        
        client = redis.from_url(settings.REDIS_URL, db=settings.REDIS_DB)
        client.ping()
        
        info = client.info()
        print(f"  ✓ Redis connected")
        print(f"  ℹ Redis version: {info.get('redis_version', 'unknown')}")
        print(f"  ℹ Used memory: {info.get('used_memory_human', 'unknown')}")
        
        print("\n✅ Redis connection successful\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Redis connection failed: {e}")
        print("\n❌ Redis not available")
        print("Start Redis with: docker run -d -p 6379:6379 redis:7-alpine")
        return False


def test_openai_connection():
    """Test OpenAI API connection"""
    print("🔍 Testing OpenAI connection...")
    
    try:
        import openai
        from app.core.config import settings
        
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Simple API test
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5
        )
        
        print(f"  ✓ OpenAI API connected")
        print(f"  ℹ Response: {response.choices[0].message.content}")
        
        print("\n✅ OpenAI connection successful\n")
        return True
        
    except Exception as e:
        print(f"  ✗ OpenAI connection failed: {e}")
        print("\n❌ OpenAI API not accessible")
        print("Check your API key in .env file")
        return False


def test_vector_db():
    """Test ChromaDB initialization"""
    print("🔍 Testing Vector Database...")
    
    try:
        from app.vectordb.manager import vector_db
        
        stats = vector_db.get_collection_stats()
        print(f"  ✓ ChromaDB initialized")
        print(f"  ℹ Collection: {stats.get('collection_name')}")
        print(f"  ℹ Total chunks: {stats.get('total_chunks', 0)}")
        
        print("\n✅ Vector Database ready\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Vector DB initialization failed: {e}")
        print("\n❌ ChromaDB not ready")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 Krishi AI Backend - System Check")
    print("="*60 + "\n")
    
    results = []
    
    # Test 1: Package imports
    results.append(("Package Imports", test_imports()))
    
    # Test 2: Environment config
    results.append(("Environment Config", test_env_config()))
    
    # Test 3: Redis
    results.append(("Redis Connection", test_redis_connection()))
    
    # Test 4: OpenAI (skip if env not configured)
    if results[1][1]:  # Only if env config passed
        results.append(("OpenAI Connection", test_openai_connection()))
    
    # Test 5: Vector DB
    results.append(("Vector Database", test_vector_db()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! System is ready.")
        print("\nStart the server with:")
        print("  uvicorn main:app --reload")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
