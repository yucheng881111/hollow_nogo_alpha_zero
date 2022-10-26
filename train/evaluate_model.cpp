#include <bits/stdc++.h>
#include <torch/torch.h>
#include "neural/network.h"


int main() {

	auto tmp = torch::zeros({1, 7, 9, 9});

	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			tmp.index_put_({0, 6, i, j}, 1);
		}
	}
	
	int channel_size = 7;
	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	// planes, height, width, filters, num_res_blocks, policy_size
	const auto net_op = az::NetworkOptions{channel_size, 9, 9, 128, 2, 81};
	az::AlphaZeroNetwork net(net_op);
	torch::load(net, "weights.pt");
	net->to(device);
	tmp = tmp.to(device);
	net->eval();
	auto [v_out, p_out] = net->forward(tmp);

	std::cout << tmp << std::endl;

	float v_pred = v_out.view(-1)[0].template item<float>();
	std::cout << "value: " << v_pred << std::endl << std::endl;
	p_out = torch::softmax(p_out, 1);

	for (int j = 0; j < 81; ++j) {
		float prob = p_out[0][j].template item<float>();
		std::cout << "action: " << j << " prob: " << prob << std::endl;
	}

return 0;	
}






