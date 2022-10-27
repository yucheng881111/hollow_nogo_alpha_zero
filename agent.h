/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include <fstream>
#include <queue>
#include <torch/torch.h>
#include "neural/network.h"
#include "stateTorch.h"

//#define DEBUG


class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b, int result) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
		else
			engine.seed((unsigned)time(NULL));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown N=0 " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}
	/*
	virtual action take_action(const board& state) {
		int N = meta["N"];
		if(N){
			node* root = new node(state);
			int result = root->MCTS(N, engine);
			delete_tree(root);
			if(result != -1){
				return action::place(result, state.info().who_take_turns);
			}else{
				return action();
			}
		}

		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}
	*/
	virtual action take_action(const board& state, int result) {
		return action::place(result, state.info().who_take_turns);
	}


	class node : board {
	public:
		int win_cnt;
		int total_cnt;
		int place_pos;
		//std::vector<node*> child;
		std::unordered_map<int, node*> child;
		node* parent;

		node(const board& state, int m = -1): board(state), place_pos(m), win_cnt(0), total_cnt(0), parent(nullptr) {}

		float win_rate(){
			if(win_cnt == 0 && total_cnt == 0){
				return 0.0;
			}
			
			return (float)win_cnt / total_cnt;
		}

		float ucb(){
			float c = 0.5;

			if(parent->total_cnt == 0 || total_cnt == 0){
				return win_rate();
			}

			return win_rate() + c * std::sqrt(std::log(parent->total_cnt) / total_cnt);
		}

		float ucb_opponent(){
			float c = 0.5;

			if(parent->total_cnt == 0 || total_cnt == 0){
				return 1 - win_rate();
			}
			
			return (1 - win_rate()) + c * std::sqrt(std::log(parent->total_cnt) / total_cnt);
		}

		int MCTS(int N, std::default_random_engine& engine){
			// 1. select  2. expand  3. simulate  4. back propagate
			
			for(int i = 0; i < N; ++i){
				// debug
				//std::fstream debug("record.txt", std::ios::app);

				// select
				//debug << "select" << std::endl;
				std::vector<node*> path = select_root_to_leaf(info().who_take_turns, engine);
				// expand
				//debug << "expand" << std::endl;
				node* leaf = path.back();
				node* expand_node = leaf->expand_from_leaf(engine);
				if(expand_node != leaf){
					path.push_back(expand_node);
				}
				// simulate
				//debug << "simulate" << std::endl;
				unsigned winner = path.back()->simulate_winner(engine);
				// backpropagate
				//debug << "backpropagate" << std::endl;
				back_propagate(path, winner);

				//debug.close();
			}

			return select_action();
		}

		int select_action(){
			// select child node who has the highest win rate (highest Q)
			if(child.size() == 0){
				return -1;
			}

			float max_score = -std::numeric_limits<float>::max();
			node* c;
			for(auto &ch : child){
				float tmp = ch.second->win_rate();
				if(tmp > max_score){
					max_score = tmp;
					c = ch.second;
				}
			}
			
			return c->place_pos;
		}

		std::vector<node*> select_root_to_leaf(unsigned who, std::default_random_engine& engine){
			std::vector<node*> vec;
			node* curr = this;
			vec.push_back(curr);

			while(!curr->is_leaf()){
				// select node who has the highest ucb score
				float max_score = -std::numeric_limits<float>::max();
				node* c;
				if(curr->child.size() == 0){
					break;
				}
				
				for(auto &curr_child : curr->child){
					float tmp;
					if(who == curr->info().who_take_turns){
						tmp = curr_child.second->ucb();
					}else{
						tmp = curr_child.second->ucb_opponent();
					}
					if(tmp > max_score){
						max_score = tmp;
						c = curr_child.second;
					}

				}
				
				vec.push_back(c);
				curr = c;
			}

			return vec;
		}

		bool is_leaf(){
			int cnt = 0;
			for(int i = 0; i < 81; i++){
				if(board(*this).place(i) == board::legal){
					cnt++;
				}
			}
			// check if fully expanded (leaf == not fully expanded)
			return !(cnt > 0 && child.size() == cnt);
		}

		node* expand_from_leaf(std::default_random_engine& engine){
			board b;
			std::vector<int> vec = all_space(engine);
			bool success_placed = 0;
			int pos = -1;
			
			for(int i = 0; i < vec.size(); ++i){
				b = *this;
				if(b.place(vec[i]) == board::legal && (*this).child.count(vec[i]) == 0){
					pos = vec[i];
					success_placed = 1;
					break;
				}
			}

			if(success_placed){
				node* new_node = new node(b, pos);
				//this->child.push_back(new_node);
				this->child[pos] = new_node;
				new_node->parent = this;
				return new_node;
			}else{
				return this;
			}
		}

		unsigned simulate_winner(std::default_random_engine& engine){
			board b = *this;
			std::vector<int> vec = all_space(engine);
			std::queue<int> q;
			for(int i = 0; i < vec.size(); ++i){
				q.push(vec[i]);
			}

			int cnt = 0;
			while(cnt != q.size()){
				int i = q.front();
				q.pop();
				if(b.place(i) != board::legal){
					q.push(i);
					cnt++;
				}else{
					cnt = 0;
				}
			}

			if(b.info().who_take_turns == board::white){
				return board::black;
			}else{
				return board::white;
			}
		}

		std::vector<int> all_space(std::default_random_engine& engine){
			std::vector<int> vec;
			for(int i = 0; i < 81; ++i){
				vec.push_back(i);
			}
			std::shuffle(vec.begin(), vec.end(), engine);
			return vec;
		}

		void back_propagate(std::vector<node*>& path, unsigned winner){
			for(int i = 0; i < path.size(); ++i){
				path[i]->total_cnt++;
				if(winner == info().who_take_turns){
					path[i]->win_cnt++;
				}
			}
		}
	};

	void delete_tree(node* root){
		if(root->child.size() == 0){
			delete root;
			return ;
		}

		for(auto &c : root->child){
			if(c.second != nullptr){
				delete_tree(c.second);
			}
		}

		delete root;
	}


private:
	std::vector<action::place> space;
	board::piece_type who;
};


class TreeNode {
public:
	TreeNode* parent = nullptr;
	std::unordered_map<int, TreeNode*> children;
	int n_visits = 0;
	float Q = 0;
	float u = 0;
	float P = 0;
	TreeNode(TreeNode* par, float p) : parent(par), P(p), u(p) {}

	void expand(std::unordered_map<int, float> &m) {
		for (auto &tmp : m) {
			int action = tmp.first;
			float prob = tmp.second;
			children[action] = new TreeNode(this, prob);
		}
	}

	std::pair<int, TreeNode*> select() {
		int best_act;
		TreeNode* best_node;
		float max_value = -std::numeric_limits<float>::max();
		for (auto &child : children) {
			if ((child.second)->get_value() > max_value) {
				max_value = (child.second)->get_value();
				best_act = child.first;
				best_node = child.second;
			}
		}

		return std::make_pair(best_act, best_node);
	}

	float get_value() {
		return Q + u;
	}

	bool is_leaf() {
		return children.empty();
	}

	bool is_root() {
		return parent == nullptr;
	}

	void update(float leaf_value, float c_puct) {
		#ifdef DEBUG
			std::fstream debug("record.txt", std::ios::app);
		#endif

		n_visits++;
		Q += ((leaf_value - Q) / n_visits);

		if (!is_root()) {
			u = c_puct * P * std::sqrt(parent->n_visits) / (1 + n_visits);
			#ifdef DEBUG
				debug << "parent_n_visits: " << parent->n_visits << " n_visits: " << n_visits << " Q: " << Q << " u: " << u << std::endl;
			#endif
		}
		
		#ifdef DEBUG
			debug.close();
		#endif
	}

	void update_recursive(float leaf_value, float c_puct) {
		if (parent != nullptr) {
			parent->update_recursive(leaf_value, c_puct);
		}
		update(leaf_value, c_puct);
	}
};

class MCTS {
public:
	TreeNode* root = nullptr;
	float lmbda = 0.5;
	float c_puct = 5;
	int rollout_limit = 500;
	int L = 20;
	int n_playout = 10000;
	az::AlphaZeroNetwork network = nullptr;

	MCTS(float lmb, float c, int rollout, int playout_depth, int p) : lmbda(lmb), c_puct(c), rollout_limit(rollout), L(playout_depth), n_playout(p) {
		root = new TreeNode(nullptr, 1.0);
		const auto net_op = az::NetworkOptions{7, 9, 9, 128, 2, 81};
		az::AlphaZeroNetwork net(net_op);
		network = net;
		std::string weights_file = "weights.pt";
		torch::load(network, weights_file);
		network->to(torch::kCUDA);
	}

	std::pair<torch::Tensor, torch::Tensor> forward_data(torch::Tensor tmp_data) {
		tmp_data = tmp_data.to(torch::kCUDA);
		auto [v_out, p_out] = network->forward(tmp_data);
		p_out = torch::softmax(p_out, 1);

		//std::fstream debug("record.txt", std::ios::app);
		//debug << v_out << std::endl;
		//debug << p_out << std::endl;
		//debug.close();
		return std::make_pair(v_out, p_out);
	} 

	float value_fn(board seq1, board seq2, board seq3) {
		auto data = getTensor(seq1, seq2, seq3);
		std::pair<torch::Tensor, torch::Tensor> result = forward_data(data);
		auto v_out = result.first;
		float v_pred = v_out.view(-1)[0].template item<float>();
		return v_pred;
	}

	std::unordered_map<int, float> policy_fn(board seq1, board seq2, board seq3) {
		auto data = getTensor(seq1, seq2, seq3);
		std::pair<torch::Tensor, torch::Tensor> result = forward_data(data);
		auto p_out = result.second;
		std::unordered_map<int, float> action_probs;

		for (int j = 0; j < 81; ++j) {
			board curr_b = seq3;
			//debug << p_out[0][j].template item<float>() << " ";
			if (curr_b.place(j) == board::legal) {
				float prob = p_out[0][j].template item<float>();
				action_probs[j] = prob;
			}
		}

		//debug << std::endl;
		return action_probs;
	}

	int get_max_action(std::unordered_map<int, float>& action_probs) {
		int best_action;
		float best_prob = -std::numeric_limits<float>::max();
		for (auto &act_prob : action_probs) {
			if (act_prob.second > best_prob) {
				best_prob = act_prob.second;
				best_action = act_prob.first;
			}
		}
		return best_action;
	}

	void playout(int leaf_depth, board seq1, board seq2, board seq3) {
		TreeNode* node = root;
		#ifdef DEBUG
			std::fstream debug("record.txt", std::ios::app);
			debug << "select: " << std::endl;
		#endif

		for (int i = 0; i < leaf_depth; ++i) {
			if (node->is_leaf()) {
				std::unordered_map<int, float> action_probs = policy_fn(seq1, seq2, seq3);
				// if end game, break
				if (action_probs.empty()) {
					break;
				}

				node->expand(action_probs);
			}
			std::pair<int, TreeNode*> tmp = node->select();
			node = tmp.second;
			int action = tmp.first;
			seq1 = seq2;
			seq2 = seq3;
			seq3.place(action);
			#ifdef DEBUG
				debug << "action: " << action << std::endl;
			#endif
		}
		
		float v = value_fn(seq1, seq2, seq3);
		//int z = evaluate_rollout(seq1, seq2, seq3, rollout_limit);
		int z = 0;
		float leaf_value = (1-lmbda) * v + lmbda * z;

		#ifdef DEBUG
			debug << "update: " << std::endl;
		#endif

		node->update_recursive(leaf_value, c_puct);

		#ifdef DEBUG
			debug << std::endl;
			debug.close();
		#endif
	}

	int evaluate_rollout(board seq1, board seq2, board seq3, int limit) {
		unsigned player = seq3.info().who_take_turns;
		for (int i = 0; i < limit; ++i) {
			std::unordered_map<int, float> action_probs = policy_fn(seq1, seq2, seq3);
			if (action_probs.empty()) {
				break;
			}
			int max_action = get_max_action(action_probs);
			seq1 = seq2;
			seq2 = seq3;
			seq3.place(max_action);
		}

		if (player == seq3.info().who_take_turns) {
			return -1;
		} else {
			return 1;
		}
	}

	int get_move(MovingStates &moving_states) {
		board seq1 = moving_states.states[0];
		board seq2 = moving_states.states[1];
		board seq3 = moving_states.states[2];
		for (int i = 0; i < n_playout; ++i) {
			playout(L, seq1, seq2, seq3);
		}

		int best_action = -1;
		/*
		// choose child by most visit
		int most_visit = 0;
		for (auto &child : root->children) {
			if ((child.second)->n_visits > most_visit) {
				most_visit = (child.second)->n_visits;
				best_action = child.first;
			}
		}
		*/

		// choose child by best Q + u
		float best_value = -std::numeric_limits<float>::max();
		for (auto &child : root->children) {
			if ((child.second)->get_value() > best_value) {
				best_value = (child.second)->get_value();
				best_action = child.first;
			}
		}

		return best_action;
	}

};

void clean_node(TreeNode* root) {
	if (root->children.size() == 0) {
		delete root;
		return ;
	}

	for(auto &c : root->children){
		if(c.second != nullptr){
			clean_node(c.second);
		}
	}

	delete root;
}

