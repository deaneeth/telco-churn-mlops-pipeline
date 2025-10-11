All evidence can be reproduced in < 5 minutes:

# 1. Start Kafka
docker compose -f docker-compose.kafka.yml up -d

# 2. Create topics
bash scripts/kafka_create_topics.sh

# 3. Run 60-second demo
bash scripts/run_kafka_demo.sh

# 4. Extract samples
bash scripts/dump_kafka_topics.sh