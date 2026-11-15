"""审计已生成的三个Cora样本：训练输入编号与测试节点严格不相交。"""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import file_hash, save_json
from src.workflow import evaluate_checkpoint


def check_bundle(folder):
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['settings']['dataset'] != 'Cora':
        raise ValueError('当前核验只允许Cora。')
    for name, digest in metadata['sha256'].items():
        if file_hash(folder / name) != digest:
            raise ValueError('样本文件散列变化：' + name)
    training = torch.load(folder / 'training.pt', map_location='cpu')
    evaluation = torch.load(folder / 'evaluation.pt', map_location='cpu')
    indices = torch.load(folder / 'attack_training_indices.pt', map_location='cpu')
    nodes, test = indices['local_to_original'], evaluation['test_nodes']
    total = evaluation['truth'].numel()
    assert nodes.numel() + test.numel() == total
    assert not torch.isin(nodes, test).any()
    assert torch.equal(torch.cat([nodes, test]).sort().values, torch.arange(total))
    assert metadata['budget']['training_graph_nodes'] == nodes.numel()
    assert metadata['budget']['requested_trigger_groups'] == int(nodes.numel() * 3 / 100)
    assert nodes.numel() == 2167 and test.numel() == 541
    for name in ['labeled_train_local', 'validation_local', 'attached_local', 'available_local']:
        position = indices[name]
        assert not (position < 0).any() and not (position >= nodes.numel()).any()
        assert not torch.isin(nodes[position], test).any()
    assert torch.equal(nodes[indices['attached_local']], evaluation['attached'])
    assert not torch.isin(training['edges'], test).any()
    assert (evaluation['truth'][evaluation['attached']] != evaluation['target']).all()
    assert (training['labels'][test] == -1).all()
    assert evaluation['attack_test_nodes'].numel() == 271
    assert evaluation['clean_nodes'].numel() == 270
    assert not torch.isin(evaluation['attack_test_nodes'], evaluation['clean_nodes']).any()
    assert set(torch.cat([evaluation['attack_test_nodes'], evaluation['clean_nodes']]).tolist()) == set(test.tolist())
    # 测试注入只追加特征和连接，不改已有节点的特征。
    assert torch.equal(evaluation['reference_x'], evaluation['changed_x'][:training['x'].size(0)])
    assert metadata['attack_training_access']['feature_rows_passed_to_fit'] == nodes.numel()
    assert metadata['attack_training_access']['label_rows_passed_to_fit'] == nodes.numel()
    for name in ['test_features_passed_to_fit', 'test_labels_passed_to_fit', 'test_edges_passed_to_fit']:
        assert metadata['attack_training_access'][name] is False
    return {'attack': metadata['settings']['attack'], 'status': 'passed',
            'training_graph_nodes': nodes.numel(), 'labeled_train_nodes': indices['labeled_train_local'].numel(),
            'test_nodes': test.numel(), 'requested_groups': metadata['budget']['requested_trigger_groups'],
            'actual_groups': evaluation['attached'].numel(), 'test_features_or_labels_in_training_input': False}


def main():
    parser = argparse.ArgumentParser(description='Cora攻击训练信息边界与重载核验')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/strict_attack_final_check_20261115')
    args = parser.parse_args()
    reports = []
    for attack in ['GTA', 'UGBA', 'DPGBA']:
        folder = args.output_root / 'data/Cora' / attack / 'seed10'
        result = check_bundle(folder)
        model_folder = args.output_root / 'attack_checks/Cora' / attack / 'seed10/GCN'
        record = json.loads((model_folder / 'result.json').read_text(encoding='utf-8'))
        assert record['status'] == 'complete'
        metrics = evaluate_checkpoint(folder, model_folder / 'model.pt', model_folder / 'reload.json', record['arguments']['device'])
        for name in ['CA', 'ASR', 'CA_hits', 'CA_denominator', 'ASR_hits', 'ASR_denominator']:
            assert metrics[name] == record['metrics'][name], name
        result['model_reload'] = 'passed'
        reports.append(result)
        print(attack + '训练信息边界、预算和模型重载核验通过。')
    save_json(args.output_root / 'boundary_verification.json', {'status': 'passed', 'dataset': 'Cora',
        'training_epochs': 2, 'scope': '只核对接口和映射，不报告性能结论', 'results': reports})


if __name__ == '__main__':
    main()
