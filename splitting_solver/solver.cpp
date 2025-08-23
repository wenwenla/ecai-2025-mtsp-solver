#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });

                        if (this->stop && this->tasks.empty()) {
                            return;
                        }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            tasks.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return res;
    }

    void join() {
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

namespace py = pybind11;

inline float dist(float* l, float* r) {
    return sqrtf((l[0] - r[0]) * (l[0] - r[0]) + (l[1] - r[1]) * (l[1] - r[1]));
}

float solve(float* locs, int* acts, int a_len, int m) {
    float l = 0.f;
    float r = 50.f;
    while (r - l > 1e-5) {
        
        float now_length = 0.f;
        int prev_node = 0;
        int agent_id = 0;
        bool new_agent = false;
        bool additional_flag = true;

        float mid_val = (l + r) * 0.5f;
        // std::vector<bool> new_split(a_len, false);
        for (int step = 0; step < a_len; ++step) {
            float* now_loc_ptr = &locs[acts[step] * 2];
            float* prev_loc_ptr = &locs[prev_node * 2];
            float* depot_loc_ptr = &locs[0];
            float this_segment_length = dist(now_loc_ptr, prev_loc_ptr);
            float back_to_depot = dist(now_loc_ptr, depot_loc_ptr);

            new_agent = (this_segment_length + back_to_depot + now_length) > mid_val;

            // additional check
            if (now_length + dist(prev_loc_ptr, depot_loc_ptr) > mid_val) {
                additional_flag = false;
                break;
            }

            if (new_agent) {
                now_length = back_to_depot;
                ++agent_id;
                // if (step - 1 >= 0) {
                //     new_split[step - 1] = true;
                // }
            } else {
                now_length += this_segment_length;
            }
            prev_node = acts[step];
        }

        // additional check
        float* prev_loc_ptr = &locs[prev_node * 2];
        float* depot_loc_ptr = &locs[0];
        if (now_length + dist(prev_loc_ptr, depot_loc_ptr) > mid_val) {
            additional_flag = false; // not break here
        }

        if (agent_id < m && additional_flag) {
            r = mid_val;
            // final_split = std::move(new_split);
        } else {
            l = mid_val;
        }
    }
    return l; //(r + l) * 0.5f;
}

float solve_pdp(float* locs, int* acts, int a_len, int m) {
    float l = 0.f;
    float r = 50.f;
    int n_cities = a_len;
    // std::cout << "n_cities: " << n_cities << std::endl;

    while (r - l > 1e-5) {
        std::set<int> pickup_set;
        float mid_val = (l + r) * 0.5;
        int step = 0;
        int next_step = 0;
        // std::cout << "mid " <<  mid_val << std::endl;

        for (int i = 0; i < m - 1; ++i) {
            int prev_node = 0;
            step = next_step;
            std::set<int> pickup_set;
            float now_length = 0.f;
            while (true) {
                if (step >= a_len) break;
                float* now_loc_ptr = &locs[acts[step] * 2];
                float* prev_loc_ptr = &locs[prev_node * 2];
                float* depot_loc_ptr = &locs[0];
                float this_segment_length = dist(now_loc_ptr, prev_loc_ptr);
                float back_to_depot = dist(now_loc_ptr, depot_loc_ptr);
                
                if (acts[step] <= n_cities / 2) { // pickup act
                    pickup_set.insert(acts[step] - 1);
                } else {
                    pickup_set.erase((acts[step] - 1) % (n_cities / 2));
                }
                now_length += this_segment_length;
                if (now_length + back_to_depot <= mid_val) {
                    if (pickup_set.empty()) next_step = step + 1;
                } else {
                    break;
                }
                prev_node = acts[step];
                ++step;
            }
        }
        float last_seg_cost = 0;
        int prev_node = 0;
        for (int i = next_step; i < a_len; ++i) {
            float* now_loc_ptr = &locs[acts[i] * 2];
            float* prev_loc_ptr = &locs[prev_node * 2];
            float* depot_loc_ptr = &locs[0];
            float this_segment_length = dist(now_loc_ptr, prev_loc_ptr);
            float back_to_depot = dist(now_loc_ptr, depot_loc_ptr);
            last_seg_cost += this_segment_length;
            if (i == a_len - 1) {
                last_seg_cost += back_to_depot;
            }
            prev_node = acts[i];
        }
        // std::cout << next_step << " " << last_seg_cost << std::endl;
        if (last_seg_cost <= mid_val) {
            r = mid_val;
        } else {
            l = mid_val;
        }

    }
    return (r + l) * 0.5f;
}


py::array_t<float> get_minmax_length(py::array_t<float> locs, py::array_t<int> acts, py::array_t<int> num_agents) {
    ThreadPool pool(32);

    auto locs_buf = locs.request();
    auto acts_buf = acts.request();
    auto num_agents_buf = num_agents.request();
    
    std::vector<ssize_t> shape = locs_buf.shape;
    int batch_size = shape[0];
    int n_nodes = shape[1];

    auto a_shape = acts_buf.shape;
    int a_len = a_shape[1];

    float* locs_ptr = static_cast<float*>(locs_buf.ptr);
    int* acts_ptr = static_cast<int*>(acts_buf.ptr);
    int* num_agents_ptr = static_cast<int*>(num_agents_buf.ptr);

    auto result = py::array_t<float>(batch_size);
    float* out_ptr = static_cast<float*>(result.request().ptr);

    std::vector<std::shared_future<float>> futures;
    for (int i = 0; i < batch_size; ++i) {
        futures.push_back(pool.enqueue(solve, &locs_ptr[i * n_nodes * 2], &acts_ptr[i * a_len], a_len, num_agents_ptr[i]));
    }

    for (int i = 0; i < batch_size; ++i) {
        out_ptr[i] = futures[i].get(); // solve(&locs_ptr[i * n_nodes * 2], &acts_ptr[i * (n_nodes - 1)], n_nodes, m);
    }
    return result;
}

py::array_t<float> get_minmax_length_pdp(py::array_t<float> locs, py::array_t<int> acts, py::array_t<int> num_agents) {
    ThreadPool pool(32);

    auto locs_buf = locs.request();
    auto acts_buf = acts.request();
    auto num_agents_buf = num_agents.request();
    
    std::vector<ssize_t> shape = locs_buf.shape;
    int batch_size = shape[0];
    int n_nodes = shape[1];

    auto a_shape = acts_buf.shape;
    int a_len = a_shape[1];

    float* locs_ptr = static_cast<float*>(locs_buf.ptr);
    int* acts_ptr = static_cast<int*>(acts_buf.ptr);
    int* num_agents_ptr = static_cast<int*>(num_agents_buf.ptr);

    auto result = py::array_t<float>(batch_size);
    float* out_ptr = static_cast<float*>(result.request().ptr);

    std::vector<std::shared_future<float>> futures;
    for (int i = 0; i < batch_size; ++i) {
        futures.push_back(pool.enqueue(solve_pdp, &locs_ptr[i * n_nodes * 2], &acts_ptr[i * a_len], a_len, num_agents_ptr[i]));
    }

    for (int i = 0; i < batch_size; ++i) {
        out_ptr[i] = futures[i].get(); // solve(&locs_ptr[i * n_nodes * 2], &acts_ptr[i * (n_nodes - 1)], n_nodes, m);
    }
    return result;
}

PYBIND11_MODULE(splitting_solver, m) {
    m.def("get_minmax_length", &get_minmax_length, "A function implemented in C++");
    m.def("get_minmax_length_pdp", &get_minmax_length_pdp, "C++ PDP");
}