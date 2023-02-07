#include <atomic>
#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <cmath>
#include <random>
#include <thread>
#include <condition_variable>
#include <limits>
#include <chrono>

template <int b = 4>
struct state {
    using player = bool;

    enum struct status_t {
        P, W, D, ERR
    };

    std::array<uint8_t, 14> fields = {b, b, b, b, b, b, 0, b, b, b, b, b, b, 0};
    player p = 0;
    status_t status = status_t::P; 

    
    void print() const {
        std::cout << "   "; 
        for (uint8_t f = 12; f >= 7; --f) {
            std::cout << (fields[f]<10?" ":"") << int(fields[f]) << " ";
        }
        std::cout << "\n";

        std::cout << (fields[13]<10?" ":"") << int(fields[13]) << "                   ";
        std::cout << (fields[6]<10?" ":"") << int(fields[6]) << "\n";

        std::cout << "   "; 
        for (uint8_t f = 0; f < 6; ++f) {
            std::cout << (fields[f]<10?" ":"") << int(fields[f]) << " ";
        }
        std::cout << "  \n" << std::endl;
    }

    bool is_valid(uint8_t field) const {
        return (field<6 && fields[7*p+field]);
    }

    void print_valid() const {
        for (uint8_t i = 0; i<6; ++i) {
            std::cout << is_valid(i) << " ";
        }
        std::cout << std::endl;
    }

    uint8_t get_valid() const {
        for (uint8_t i = 0; i<6; ++i) {
            if (fields[7*p+i])
                return i;
        }
        return 6;
    }

    bool move(uint8_t f) {

        if (status!=status_t::P || f>5) return false;

        f += 7*p;
        uint8_t beans = fields[f];
        
        if (beans == 0) return false;

        fields[f] = 0;
        while (beans>0) {
            f = (f+1)%14;
            if (f == (!p*7+6)) continue;
            fields[f] += 1;
            beans -= 1;
        }

        if (f != 7*p+6) {
            if (f/7==p && fields[12-f] && fields[f]==1) {
                fields[7*p+6] += fields[12-f]+1;
                fields[f] = 0;
                fields[12-f] = 0;
            }
            p = !p;
        }

        check_winner();

        return true;
    }

    const uint8_t* move_mask() const {
        return fields.data()+p*7;
    }

    private :
    void check_winner() {
        uint8_t sum0 = 0, sum1 = 0;
        for (uint8_t f = 0; f<6; ++f) sum0 += fields[f];
        for (uint8_t f = 7; f<13; ++f) sum1 += fields[f];
        if (sum0 == 0 || sum1 == 0) {
            if (sum0+fields[6]>sum1+fields[13]) {
                status = status_t::W;
                p=0;
            }
            else if (sum0+fields[6]<sum1+fields[13]) {
                status = status_t::W;
                p=1;
            }
            else {
                status = status_t::D;
            }
            sum0+=fields[6];
            sum1+=fields[13];
            fields.fill(0);
            fields[6]=sum0;
            fields[13]=sum1;
        }
    }

};

using Board = state<4>;

struct Node {
    std::shared_mutex mtx;
    struct statistics {uint32_t total, p0score;};
    std::atomic<statistics> stats;
    
    std::vector<Node> children;
    Board board;
    std::atomic<uint8_t> expansion;
    uint8_t last_move = 255;
    
    Node() : stats({0,0}), expansion(0) {
        // children.reserve(10);
    }
    Node(const Node& other, uint8_t move) : stats({0,0}), board(other.board), expansion(0), last_move(move) {
        board.move(move);
        // children.reserve(10);
    }

    Node(Node&& other) : mtx(), stats(other.stats.load()), children(std::move(other.children)),
                         board(std::move(other.board)), expansion(other.expansion.load()),
                         last_move(other.last_move) {
                            // children.reserve(10);
                         }

    // ~Node() {
    //     std::cerr << "destroyed" << std::endl;
    // }
};

struct Game {

    std::unique_ptr<Node> tree = std::make_unique<Node>();
    static thread_local std::mt19937_64 rng;

    void traverse(Node*& c, std::vector<uint8_t>& path) const {
        while (c->expansion == 255) {
            Node::statistics parent_stats = c->stats;
            uint8_t max_idx = 0;
            double max_ucb = 0.0;
            if (parent_stats.total == 0) {
                max_idx = std::uniform_int_distribution<>(0, c->children.size()-1)(rng);
            } else {
                for (uint8_t i = 0; i<c->children.size(); ++i) {
                    Node* child = &(c->children[i]);
                    Node::statistics child_stats = child->stats;
                    if (child_stats.total == 0) {
                        max_idx = i;
                        break;
                    }
                    double ucb = double(c->board.p?child_stats.total-child_stats.p0score:child_stats.p0score)
                                 / child_stats.total
                                 + std::sqrt(2*std::log(parent_stats.total)/child_stats.total);
                    if (ucb>max_ucb) {
                        max_ucb = ucb;
                        max_idx = i;
                    }
                }
            }
            c = &(c->children[max_idx]);
            path.emplace_back(max_idx);
        }
    }

    uint8_t select() const {
        uint32_t max = 0;
        uint8_t max_idx = 0;
        for (uint8_t i = 0; i<tree->children.size(); ++i) {
            uint32_t visits = tree->children[i].stats.load().total;
            if (visits>max) {
                max = visits;
                max_idx = i;
            }
        }
        // std::cerr << "SELECTED: " << max << std::endl;
        return max_idx;
    }

    void move() {
        uint8_t best = select();
        auto tmp = std::make_unique<Node>(std::move(tree->children[best]));
        tree = std::move(tmp);
    }

    void move(uint8_t selection) {
        if (!tree->board.is_valid(selection))
            return;

        uint8_t move_idx = 0;
        for (uint8_t i = 0; i<=selection; i++)
            if (tree->board.move_mask()[i]) ++move_idx;
        --move_idx;

        if (tree->children.size() > move_idx) {
            auto tmp = std::make_unique<Node>(std::move(tree->children[move_idx]));
            tree = std::move(tmp);
        } else {
            tree = std::make_unique<Node>(*tree, selection);
        }
    }

    void backpropagate(uint8_t result, std::vector<uint8_t>& path) {
        Node* cur = tree.get();
        Node::statistics expected = cur->stats;
        while(!cur->stats.compare_exchange_weak(expected, {expected.total+1, expected.p0score+result}));
        
        for (uint8_t selection: path) {
            if (cur->expansion==255) {
                cur = &(cur->children[selection]);
                Node::statistics expected = cur->stats;
                while(!cur->stats.compare_exchange_weak(expected, {expected.total+1, expected.p0score+result}));
            } else {
                std::shared_lock lg(cur->mtx);
                cur = &(cur->children[selection]);
                Node::statistics expected = cur->stats;
                while(!cur->stats.compare_exchange_weak(expected, {expected.total+1, expected.p0score+result}));
            }
        }
    }

    uint8_t random_rollout(Board b) {
        while (b.status==Board::status_t::P) {
            std::array<uint8_t, 6> choices;
            uint8_t size = 0;
            for (uint8_t i = 0; i < 6; ++i) {
                if (b.move_mask()[i]) {
                    choices[size] = i;
                    ++size;
                }
            }
            b.move(choices[std::uniform_int_distribution(0,size-1)(rng)]);
        }
        switch (b.status) {
            case(Board::status_t::W):
                return !b.p;
            case(Board::status_t::D):
                return std::uniform_int_distribution(0,1)(rng);
            default:
                return 0; // should never happen
        }
    }
    
    void explore() {
        // selection
        Node* cur = tree.get();
        std::vector<uint8_t> path;
        uint8_t result = 0;
        
        // repeat selection if necessary (other thread undercuts expansion)
        bool done = false;
        while(!done) {
            // SELECTION
            traverse(cur, path); // traverse to next unexpanded node
            
            // check reason for non-expansion
            switch (cur->board.status) {
            case(Board::status_t::P): // node just wasn't expanded yet
                {
                    Board rollout;
                    {
                        std::unique_lock lg(cur->mtx);
                        // std::cerr << "acquired lock" << std::endl;
                        uint8_t next = cur->expansion;
                        if (next==255) continue;

                        // incase expansion has never been set
                        while (!cur->board.move_mask()[next]) ++next;

                        cur->children.emplace_back(*cur, next);
                        rollout = cur->children.back().board;
                        path.emplace_back(cur->children.size()-1);

                        for(++next; next<6 && !cur->board.move_mask()[next]; ++next);
                        if (next == 6) next = 255;
                        cur->expansion = next;
                    }
                    result = random_rollout(rollout);
                }
                break;
            case(Board::status_t::W): // node is terminal (win)
                result = !cur->board.p;
                break;
            case(Board::status_t::D): // node is terminal (draw)
                result = std::uniform_int_distribution(0,1)(rng);
                break;
            default:
                // should never be reached
                break;
            }
            done = true;
        }
        backpropagate(result, path);
    }
};

thread_local std::mt19937_64 Game::rng;

constexpr size_t NUM_THREADS = 16;
constexpr size_t NUM_ITERATIONS = (1<<22) * std::min(1., NUM_THREADS/16.);
constexpr size_t MAX_ITERATIONS = 1<<23;


template<typename T>
constexpr T SDIV(T lhs, T rhs) {
    return (lhs+rhs-1)/rhs;
}


struct Control {
    std::mutex mx;
    std::condition_variable cv_workers, cv_main;
    std::atomic<size_t> pending = 0;
    size_t active = NUM_THREADS;
    enum class mode {
        WORK, STOP
    };
    std::atomic<mode> mode = Control::mode::WORK;
    size_t cycle = 1;
};


void job(size_t p, Game& game, Control& control) {
    size_t _cycle = 0;
    while (control.mode != Control::mode::STOP) {
        size_t _pending = control.pending;
        if (_pending > 0) {
            if (control.pending.compare_exchange_weak(_pending, _pending-1)) {
                game.explore();
            }
        } else {
            ++_cycle;
            std::unique_lock lock(control.mx);
            --control.active;
            // std::cerr << "active: " << control.active << std::endl;
            if (control.active == 0) {
                lock.unlock(); // unlock manually so main doesn't need to wait
                control.cv_main.notify_one();
                
                lock.lock(); // see above
            }
            // std::cerr << "worker going to sleep" << std::endl;
            control.cv_workers.wait(lock, [&]{return control.mode==Control::mode::STOP || control.cycle!=_cycle;});
            // std::cerr << "worker waking up" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    // initialize
    Game game;
    game.tree->board.p = (argc>1 && argv[1][0]=='1');
    Control control;

    // spawn workers
    std::vector<std::thread> threads;
    for (size_t p = 0; p<NUM_THREADS; ++p) {
        threads.emplace_back(job, p, std::ref(game), std::ref(control));
    }
    {
        std::unique_lock lock(control.mx);
        control.cv_main.wait(lock, [&]{return control.active == 0;});
    }
    // using namespace std::chrono_literals;
    // std::this_thread::sleep_for(2000ms);
    while (game.tree->board.status == Board::status_t::P) {
        std::cerr << "Results: " << game.tree->stats.load().total << '\n' << std::endl;
        // std::cerr << std::endl;
        // auto parent_stats = game.tree->stats.load();
        // for (const auto& i: game.tree->children) {
        //     auto child_stats = i->stats.load();
        //     std::cerr << "[" << child_stats.p0score << " " << child_stats.total << " " << double(game.tree->board.p?child_stats.total-child_stats.p0score:child_stats.p0score) / child_stats.total << " " << std::sqrt(2*std::log(parent_stats.total)/child_stats.total) 
        //     << " " << double(game.tree->board.p?child_stats.total-child_stats.p0score:child_stats.p0score) / child_stats.total + std::sqrt(2*std::log(parent_stats.total)/child_stats.total)  << "] " << std::endl;
        // }
        game.tree->board.print();
        if (game.tree->board.p) { // CPU TURN
            std::unique_lock lock(control.mx);
            control.active = NUM_THREADS;
            control.pending = NUM_ITERATIONS;
            ++control.cycle;
            
            lock.unlock();
            control.cv_workers.notify_all();
            
            lock.lock();
            control.cv_main.wait(lock, [&]{return control.active == 0;});
            
            // auto parent_stats = game.tree->stats.load();
            // for (const auto& i: game.tree->children) {
            // auto child_stats = i->stats.load();
            // std::cerr << "[" << child_stats.p0score << " " << child_stats.total << " " << double(game.tree->board.p?child_stats.total-child_stats.p0score:child_stats.p0score) / child_stats.total << " " << std::sqrt(2*std::log(parent_stats.total)/child_stats.total) 
            //           << " " << double(game.tree->board.p?child_stats.total-child_stats.p0score:child_stats.p0score) / child_stats.total + std::sqrt(2*std::log(parent_stats.total)/child_stats.total)  << "] " << std::endl;
            // }
            game.move();
            std::cout << "CPU's move: " << int(game.tree->last_move) << "\n" << std::endl;
        } else { // player's turn
            std::unique_lock lock(control.mx);
            control.active = NUM_THREADS;
            control.pending = MAX_ITERATIONS;
            ++control.cycle;
            
            lock.unlock();
            control.cv_workers.notify_all();

            int in;
            std::cout << "---------------------------\nYOUR MOVE: ";
            std::cin >> in;
            control.pending = 0;
            
            lock.lock();
            // std::cerr << "main going to sleep" << std::endl;
            control.cv_main.wait(lock, [&]{return control.active == 0;});
            // std::cerr << "main waking up" << std::endl;
            
            game.move(in);
        }
    }
    std::unique_lock lock(control.mx);
    control.mode = Control::mode::STOP;
    lock.unlock();
    control.cv_workers.notify_all();
    for (auto& t: threads)
        t.join();

    game.tree->board.print();
    std::cerr << "done." << std::endl;
}
