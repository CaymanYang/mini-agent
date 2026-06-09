#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>

using namespace std;

struct Event {
    int id;
    int priority;

    // 自定义比较逻辑：优先级高的优先；优先级相同时，ID小的优先
    bool operator<(const Event& other) const {
        if (priority != other.priority) {
            return priority < other.priority; // priority_queue 是最大堆，这里返回 true 表示 other 更大
        }
        return id > other.id; // 如果优先级相同，id 大的被认为“更小”（即在堆中排在后面）
    }
};

class EventManager {
private:
    priority_queue<Event> pq;
    unordered_map<int, int> event_map;

public:
    EventManager(vector<vector<int>>& events) {
        for (const auto& e : events) {
            int id = e[0];
            int prio = e[1];
            event_map[id] = prio;
            pq.push({id, prio});
        }
    }

    void updatePriority(int eventId, int newPriority) {
        // 题目要求：eventId 必定指向一个活跃事件
        event_map[eventId] = newPriority;
        pq.push({eventId, newPriority});
    }

    int pollHighest() {
        while (!pq.empty()) {
            Event top = pq.top();
            pq.pop();

            // 检查该记录是否是当前最新的（处理懒惰删除）
            if (event_map.count(top.id) && event_map[top.id] == top.priority) {
                event_map.erase(top.id);
                return top.id;
            }
        }
        return -1;
    }
};

void run_test(int testNum, vector<vector<int>> events, vector<pair<string, vector<int>>> queries) {
    EventManager em(events);
    cout << "Test Case " << testNum << ": " << endl;
    for (auto& q : queries) {
        if (q.first == "pollHighest") {
            cout << "pollHighest: " << em.pollHighest() << endl;
        } else if (q.first == "updatePriority") {
            em.updatePriority(q.second[0], q.second[1]);
            cout << "updatePriority: " << q.second[0] << " to " << q.second[1] << endl;
        }
    }
    cout << "--------------------------" << endl;
}

int main() {
    // 示例 1
    // Initial: [[5, 7], [2, 7], [9, 4]]
    // Queries: pollHighest, updatePriority(9, 7), pollHighest, pollHighest
    run_test(1, {{5, 7}, {2, 7}, {9, 4}}, {
        {"pollHighest", {}},
        {"updatePriority", {9, 7}},
        {"pollHighest", {}},
        {"pollHighest", {}}
    });

    // 示例 2
    // Initial: [[4, 1], [7, 2]]
    // Queries: pollHighest, pollHighest, pollHighest
    run_test(2, {{4, 1}, {7, 2}}, {
        {"pollHighest", {}},
        {"pollHighest", {}},
        {"pollHighest", {}}
    });

    return 0;
}
