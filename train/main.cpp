#include <bits/stdc++.h>
#include <torch/torch.h>
#include "neural/network.h"

//using namespace std;

std::vector<std::vector<std::vector<std::vector<bool>>>> replay_buffer;
std::vector<std::vector<int>> replay_buffer_policy;
std::vector<int> is_win;

std::vector<std::vector<std::vector<std::vector<bool>>>> replay_buffer_eval;
std::vector<std::vector<int>> replay_buffer_eval_policy;
std::vector<int> is_win_eval;

int channel_size = 3;
int batch_size = 512;
int num_of_training_data = 13000;

std::pair<int, int> reconstruct(int pos) {
	int i, j;
	if (pos >= 81) {  // white
		pos -= 81;
		j = pos / 9;
		i = 9 - (pos % 9) - 1;
		
	} else {         // black
		j = pos / 9;
		i = 9 - (pos % 9) - 1;
	}

	return {i, j};
}

void open_history() {
	srand((unsigned)time(NULL));

	std::fstream ifs("history.txt", std::ios::in);
	std::fstream ifs2("win_or_loss.txt", std::ios::in);
	
	std::string s;
	int cnt = 0;
	while (std::getline(ifs2, s)) {
        std::stringstream ss(s);
        int i;
        ss >> i;
        
        if (cnt >= num_of_training_data) {
        	is_win_eval.push_back(i); // value target
        } else {
			is_win.push_back(i); // value target
        }
		
        cnt++;
	}

	
	cnt = 0;
    while (std::getline(ifs, s)) {
		std::vector<int> vec;
        std::stringstream ss(s);
		int i;
		while (ss >> i) {
			vec.push_back(i);
		}
		
		int len = vec.size();

		std::vector<std::vector<std::vector<bool>>> trajectory;
		std::vector<std::vector<bool>> black(9, std::vector<bool>(9, 0));
		std::vector<std::vector<bool>> white(9, std::vector<bool>(9, 0));
		
		for (auto &v : vec) {
			std::pair<int, int> p = reconstruct(v);
			if (v < 81) {     // black
				black[p.first][p.second] = 1;
				trajectory.push_back(black);
			} else {          // white
				white[p.first][p.second] = 1;
				trajectory.push_back(white);
			}

		}
		
		if (cnt >= num_of_training_data) {
			replay_buffer_eval.push_back(trajectory);
			replay_buffer_eval_policy.push_back(vec);
		} else {
			replay_buffer.push_back(trajectory);
			replay_buffer_policy.push_back(vec);
		}
		
		

		cnt++;
    }

    

    ifs.close();
	ifs2.close();

}

auto Sample_Batch(int batch_size, bool eval) {
	srand((unsigned)time(NULL));

	
	auto tmp_data = torch::zeros({batch_size, channel_size, 9, 9}); // N, C, H, W
	auto tmp_p_label = torch::zeros({batch_size, 81});
	auto tmp_v_label = torch::zeros({batch_size});
	int len = replay_buffer.size();

	if (eval) {
		len = replay_buffer_eval.size();
	}

	

	for (int i = 0; i < batch_size; ++i) {
		
		int data_idx = rand() % len;
		int trajectory_idx;
		int trajectory_len;

		if (eval) {
			trajectory_len = replay_buffer_eval[data_idx].size();
		} else {
			trajectory_len = replay_buffer[data_idx].size();
		}

		while (1) {
			trajectory_idx = rand() % trajectory_len;
			if (trajectory_idx % 2 == 0 && trajectory_idx + channel_size < trajectory_len) {  // choose black for start
				break;
			}
		}

		
		for (int j = trajectory_idx; j < trajectory_idx + channel_size; ++j) {
			
			if (eval) {
				for (int m = 0; m < 9; ++m) {
					for (int n = 0; n < 9; ++n) {
						if (replay_buffer_eval[data_idx][j][m][n] == 1) {
							tmp_data.index_put_({i, j-trajectory_idx, m, n}, 1);
						}
					}
				}
			} else {
				for (int m = 0; m < 9; ++m) {
					for (int n = 0; n < 9; ++n) {
						if (replay_buffer[data_idx][j][m][n] == 1) {
							tmp_data.index_put_({i, j-trajectory_idx, m, n}, 1);
						}
					}
				}
			}
		}
		
		int p;
		if (eval) {
			p = replay_buffer_eval_policy[data_idx][trajectory_idx + channel_size]; // policy target
		} else {
			p = replay_buffer_policy[data_idx][trajectory_idx + channel_size]; // policy target
		}
		
		//tmp_p_label[i][p-81] = 1;
		tmp_p_label.index_put_({i, p-81}, 1);
		
		int v;
		if (eval) {
			v = is_win_eval[data_idx];
		} else {
			v = is_win[data_idx];
		}

		//tmp_v_label[i] = v;
		tmp_v_label.index_put_({i}, v);
		
	}
		
	//tmp_data = tmp_data.to(device);
	//tmp_p_label = tmp_p_label.to(device);
	//tmp_v_label = tmp_v_label.to(device);
	return std::make_tuple(tmp_data, tmp_p_label, tmp_v_label);
}



int main(){
	
	srand((unsigned)time(NULL));
	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	// planes, height, width, filters, num_res_blocks, policy_size
	const auto net_op = az::NetworkOptions{channel_size, 9, 9, 64, 2, 81};
	
	az::AlphaZeroNetwork net(net_op);
	net->to(device);

	open_history();

	int replay_buffer_size = replay_buffer.size();
	int replay_buffer_eval_size = replay_buffer_eval.size();
	std::cout << "replay buffer size: " << replay_buffer_size << std::endl;
	std::cout << "replay buffer eval size: " << replay_buffer_eval_size << std::endl;

	// debug
	
	//std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tp = Sample_Batch(batch_size, 0);
	//auto data = std::get<0>(tp).to(device);
	//auto p_label = std::get<1>(tp).to(device);
	//auto v_label = std::get<2>(tp).to(device);
	//std::cout << data << std::endl;
	//std::cout << p_label << std::endl;
	//std::cout << v_label << std::endl;
	

	
	int epoch = 150;
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.002).weight_decay(1e-4));

	for(int i = 0; i < epoch; ++i) {
		//net->zero_grad();
		net->train();

		float batch_total_loss_v = 0.0;
		float batch_total_loss_p = 0.0;
		int cnt_v = 0, cnt_p = 0;
		for (int k = 0; k < 1300; ++k) {
			optimizer.zero_grad();
			std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tp = Sample_Batch(batch_size, 0);

			auto data = std::get<0>(tp).to(device);
			auto p_label = std::get<1>(tp).to(device);
			auto v_label = std::get<2>(tp).to(device);

			auto [v_out, p_out] = net->forward(data);

			auto v_loss = torch::mse_loss(v_out.view(-1), v_label);
			auto p_loss = torch::mean(-torch::sum(p_label * torch::log_softmax(p_out, 1), 1));
			
			batch_total_loss_v += v_loss.template item<float>();
			batch_total_loss_p += p_loss.template item<float>();
			auto total_loss = v_loss + p_loss;
			total_loss.backward();
			optimizer.step();

			// calculate accuracy
			for (int j = 0; j < batch_size; ++j) {
				float v_truth = v_label[j].template item<float>();
				float v_pred = v_out.view(-1)[j].template item<float>();
				if (v_truth == 1 && v_pred >= 0) {
					cnt_v++;
				} else if (v_truth == -1 && v_pred < 0) {
					cnt_v++;
				}

				auto p_truth = std::get<1>(torch::max(p_label, 1))[j];
				auto p_pred = std::get<1>(torch::max(p_out, 1))[j];
				float p_truth_f = p_truth.template item<float>();
				float p_pred_f = p_pred.template item<float>();
				
				if (p_truth_f == p_pred_f) {
					cnt_p++;
				}
			}
		}
		float batch_total_loss = batch_total_loss_v + batch_total_loss_p;
		std::cout << "epoch " << i + 1 << ":\n";
		std::cout << "value acc: " << (float)cnt_v / (batch_size*1300) << "  ";
		std::cout << "policy acc: " << (float)cnt_p / (batch_size*1300) << std::endl;
		std::cout << "value loss: " << batch_total_loss_v << "  ";
		std::cout << "policy loss: " << batch_total_loss_p << "  ";
		std::cout << "total loss: " << batch_total_loss << std::endl;
		
		
		if ((i + 1) % 5 == 0) {
			std::cout << "eval:" << std::endl;
			net->eval();
			int cnt_v = 0, cnt_p = 0;
			float batch_total_loss_v = 0.0;
			float batch_total_loss_p = 0.0;
			for (int k = 0; k < 200; ++k) {
				std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tp_eval = Sample_Batch(batch_size, 1);
				
				auto data_eval = std::get<0>(tp_eval).to(device);
				auto p_label_eval = std::get<1>(tp_eval).to(device);
				auto v_label_eval = std::get<2>(tp_eval).to(device);

				auto [v_out_eval, p_out_eval] = net->forward(data_eval);

				auto v_loss = torch::mse_loss(v_out_eval.view(-1), v_label_eval);
				auto p_loss = torch::mean(-torch::sum(p_label_eval * torch::log_softmax(p_out_eval, 1), 1));
				
				batch_total_loss_v += v_loss.template item<float>();
				batch_total_loss_p += p_loss.template item<float>();
				
				//std::cout << v_label_eval << std::endl;
				//std::cout << v_out_eval.view(-1) << std::endl;
				
				for (int j = 0; j < batch_size; ++j) {
					float v_truth = v_label_eval[j].template item<float>();
					float v_pred = v_out_eval.view(-1)[j].template item<float>();
					if (v_truth == 1 && v_pred >= 0) {
						cnt_v++;
					} else if (v_truth == -1 && v_pred < 0) {
						cnt_v++;
					}

					auto p_truth = std::get<1>(torch::max(p_label_eval, 1))[j];
					auto p_pred = std::get<1>(torch::max(p_out_eval, 1))[j];
					float p_truth_f = p_truth.template item<float>();
					float p_pred_f = p_pred.template item<float>();
					
					if (p_truth_f == p_pred_f) {
						cnt_p++;
					}
					
				}
			}
			
			std::cout << "value acc: " << (float)cnt_v / (batch_size*200) << "  ";
			std::cout << "policy acc: " << (float)cnt_p / (batch_size*200) << std::endl;
			std::cout << "value loss: " << batch_total_loss_v << "  ";
			std::cout << "policy loss: " << batch_total_loss_p << "  ";
			std::cout << "total loss: " << batch_total_loss_v + batch_total_loss_p << std::endl << std::endl;

			torch::save(net, "epoch" + std::to_string(i) + "_weights.pt");
		}
		
	}

	


return 0;
}







