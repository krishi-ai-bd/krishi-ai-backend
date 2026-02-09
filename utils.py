"""
Utility script for maintenance tasks
"""
import asyncio
from app.utils.cache_manager import cache_manager
from app.vectordb.manager import vector_db


def cleanup_inactive_users():
    """Remove inactive users and their conversations"""
    print("🧹 Starting cleanup of inactive users...")
    cache_manager.cleanup_inactive_users()
    print("✓ Cleanup complete!")


def get_stats():
    """Display system statistics"""
    print("\n📊 System Statistics")
    print("=" * 50)
    
    # Vector DB stats
    vdb_stats = vector_db.get_collection_stats()
    print(f"Vector DB Chunks: {vdb_stats.get('total_chunks', 0)}")
    
    # Redis stats
    if cache_manager.redis_client:
        try:
            info = cache_manager.redis_client.info()
            print(f"Redis Connected Clients: {info.get('connected_clients', 'N/A')}")
            print(f"Redis Used Memory: {info.get('used_memory_human', 'N/A')}")
            
            # Count keys
            user_keys = cache_manager.redis_client.scan_iter(match="user:*")
            chat_keys = cache_manager.redis_client.scan_iter(match="chat:*")
            
            user_count = sum(1 for _ in user_keys)
            chat_count = sum(1 for _ in chat_keys)
            
            print(f"Active Users: {user_count}")
            print(f"Total Conversations: {chat_count}")
        except Exception as e:
            print(f"Redis Stats Error: {e}")
    else:
        print("Redis: Not connected")
    
    print("=" * 50)


def delete_user_data(user_id: str):
    """Delete all data for a specific user"""
    confirm = input(f"⚠️  Delete all data for user '{user_id}'? (yes/no): ")
    if confirm.lower() == 'yes':
        cache_manager.delete_user_data(user_id)
        print(f"✓ Deleted data for user: {user_id}")
    else:
        print("Cancelled")


def delete_document(document_name: str):
    """Delete a document from vector DB"""
    confirm = input(f"⚠️  Delete document '{document_name}'? (yes/no): ")
    if confirm.lower() == 'yes':
        vector_db.delete_by_document(document_name)
        print(f"✓ Deleted document: {document_name}")
    else:
        print("Cancelled")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n🛠️  Krishi AI Maintenance Utilities")
        print("\nUsage:")
        print("  python utils.py stats              - Show system statistics")
        print("  python utils.py cleanup            - Remove inactive users")
        print("  python utils.py delete_user <id>   - Delete specific user data")
        print("  python utils.py delete_doc <name>  - Delete specific document")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "stats":
        get_stats()
    elif command == "cleanup":
        cleanup_inactive_users()
    elif command == "delete_user" and len(sys.argv) > 2:
        delete_user_data(sys.argv[2])
    elif command == "delete_doc" and len(sys.argv) > 2:
        delete_document(sys.argv[2])
    else:
        print("❌ Invalid command or missing arguments")
