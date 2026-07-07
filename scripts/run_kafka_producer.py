import sys
import json
import random
import time
import argparse
from datetime import datetime
from run_gap_filler import StatefulEcommerceGenerator

def start_streaming(generator, bootstrap_servers, delay):
    try:
        from kafka import KafkaProducer
    except ImportError:
        print("❌ Error: kafka-python-ng is not installed. Please run: pip install kafka-python-ng")
        sys.exit(1)
        
    print(f"🔌 Connecting to Kafka broker at {bootstrap_servers}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            max_block_ms=5000
        )
        print("✅ Connected to Kafka successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        print("Please check if your Docker containers are running (docker compose ps).")
        sys.exit(1)
        
    # We keep a rolling cache of recent purchases to review in real-time
    recent_purchased = list(generator.purchased_items[-100:]) if generator.purchased_items else []
    
    last_user_added_day = datetime.now().date()
    users_to_add_today = random.randint(0, 200)
    users_added_today = 0
    
    print(f"🚀 Streaming simulator started in real-time. Daily user quota for today: {users_to_add_today}")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            now = datetime.now()
            now_str = now.isoformat()
            today = now.date()
            
            if today != last_user_added_day:
                last_user_added_day = today
                users_to_add_today = random.randint(0, 200)
                users_added_today = 0
                print(f"📅 New day detected! User quota for {today}: {users_to_add_today}")
                
            roll = random.random()
            
            # User Registration (if quota remains)
            if roll < 0.10 and users_added_today < users_to_add_today:
                user = generator.generate_user(today.strftime("%Y-%m-%d"))
                users_added_today += 1
                
                producer.send("db_users", user)
                print(f"👤 USER SIGNUP: {user['user_id']} in {user['city']} (Today: {users_added_today}/{users_to_add_today})")
                
            # User Event (View/Cart)
            elif roll < 0.70:
                if generator.active_users:
                    user_id = random.choice(generator.active_users)
                    event = generator.generate_event(user_id, now_str)
                    
                    producer.send("db_events", event)
                    print(f"👁️ EVENT: {user_id} {event['event_type'].upper()} {event['product_id']}")
                    
            # User Checkout (Order + Order Items)
            elif roll < 0.90:
                if generator.active_users:
                    user_id = random.choice(generator.active_users)
                    order, items = generator.generate_order_and_items(user_id, now_str)
                    
                    for item in items:
                        producer.send("db_order_items", item)
                        recent_purchased.append(item)
                        if len(recent_purchased) > 200:
                            recent_purchased.pop(0)
                            
                    producer.send("db_orders", order)
                    print(f"🛍️ ORDER PLACED: {order['order_id']} by {user_id} - {len(items)} items - Total ${order['total_amount']:.2f}")
                    
            # User Review
            else:
                if recent_purchased:
                    purchase = random.choice(recent_purchased)
                    review = generator.generate_review(purchase, now_str)
                    
                    producer.send("db_reviews", review)
                    print(f"⭐ REVIEW: {purchase['user_id']} rated {purchase['product_id']} {review['rating']} stars: \"{review['review_text']}\"")
                    recent_purchased.remove(purchase)
                    
            if random.random() < 0.20:
                producer.flush()
                
            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\n👋 Streaming simulator stopped by user.")
        producer.flush()
        producer.close()

def main():
    parser = argparse.ArgumentParser(description="E-Commerce Real-time Kafka Producer")
    parser.add_argument(
        "--bootstrap-servers", 
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=2.0,
        help="Delay between streamed events in seconds (default: 2.0)"
    )
    args = parser.parse_args()
    
    print("📋 Initializing generator and loading baseline data...")
    generator = StatefulEcommerceGenerator()
    print(f"📦 Loaded {len(generator.products)} products, {len(generator.cities)} cities, {len(generator.active_users)} active users.")
    
    start_streaming(generator, args.bootstrap_servers, args.delay)

if __name__ == "__main__":
    main()
