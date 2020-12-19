import dimod
import minorminer
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
import dwave.embedding
import pandas as pd
dict_qpus = {
    "Advantage": "Advantage_system1.1",
    "2000Q": "DW_2000Q_6"
}


def _get_dwave_sampler(solver_type="Advantage") -> DWaveSampler:
    """D-Wave Samplerを取得

    Args:
        solver_type (str, optional): 
            D-WaveマシンをAdvantageもしくは2000Qのどちらかを使う.
            Defaults to "Advantage".

    Raises:
        ValueError: solver_typeが"Advantage"と"2000Q"以外だった場合にエラーを出す

    Returns:
        DWaveSampler: sampelerを返す
    """

    if solver_type not in dict_qpus:
        raise ValueError()
    # ".dwavekey"ファイルを作成し、dwave tokenを記入する
    with open(".dwavekey", "r") as f:
        token: str = f.read()
    endpoint: str = 'https://cloud.dwavesys.com/sapi'
    sampler: DWaveSampler = DWaveSampler(
        endpoint=endpoint,
        solver=dict_qpus[solver_type], token=token
    )
    return sampler


def generate_random_adder_qubo(n_digits: int) -> dimod.binary_quadratic_model:
    """QUBOを生成する

    Args:
        n_digits (int): A,Bの桁数を指定します

    Returns:
        dimod.binary_quadratic_model: QUBO
    """
    dict_qubo: dict = {}
    for i in range(n_digits):
        dict_qubo[(f"A{i}", f"A{i}")] = 0.2
        dict_qubo[(f"B{i}", f"B{i}")] = 0.2
        dict_qubo[(f"s{i+1}", f"s{i+1}")] = 0.8
        dict_qubo[(f"C{i}", f"C{i}")] = 0.2
        dict_qubo[(f"A{i}", f"B{i}")] = 0.4
        dict_qubo[(f"A{i}", f"C{i}")] = -0.4
        dict_qubo[(f"B{i}", f"C{i}")] = -0.4
        dict_qubo[(f"A{i}", f"s{i+1}")] = -0.8
        dict_qubo[(f"B{i}", f"s{i+1}")] = -0.8
        dict_qubo[(f"C{i}", f"s{i+1}")] = 0.8
        if i != 0:
            dict_qubo[(f"s{i}", f"s{i}")] += 0.2
            dict_qubo[(f"A{i}", f"s{i}")] = 0.4
            dict_qubo[(f"B{i}", f"s{i}")] = 0.4
            dict_qubo[(f"C{i}", f"s{i}")] = -0.4
            dict_qubo[(f"s{i}", f"s{i+1}")] = -0.8
    bqm = dimod.BinaryQuadraticModel.from_qubo(dict_qubo)
    return bqm


def solve_dwave_mineng(
        bqm: dimod.binary_quadratic_model, solver_type="Advantage", num_reads=1000
) -> pd.DataFrame:
    """Dwaveでquboを解く

    Args:
        bqm (dimod.binary_quadratic_model): QUBOを入力します
        solver_type (str, optional): solverのタイプを選ぶ. Defaults to "Advantage".
        num_reads (int, optional): Dwaveでの計算回数をしています. Defaults to 1000.
    Returns:
        pd.DataFrame: 結果を返します
    """
    sampler: DWaveSampler = _get_dwave_sampler(solver_type=solver_type)
    embedding = minorminer.find_embedding(bqm.quadratic, sampler.edgelist)
    cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)
    composite = FixedEmbeddingComposite(sampler, embedding)
    result = composite.sample(
        bqm, chain_strength=0.5, num_reads=num_reads,
        auto_scale=True, chain_break_method=cbm)
    df_result = result.to_pandas_dataframe()
    return df_result


def result_summary(df_result: pd.DataFrame, n_digits: int, is_correct=True, is_print_result=True):
    df_index = df_result.index
    if is_correct:
        df_index = df_index[df_result.energy < 1e-5]
    for i in df_index:
        tmp_result = df_result.loc[i]
        str_a_bin = ""
        str_b_bin = ""
        str_c_bin = ""
        for j in range(n_digits):
            str_a_bin = str(int(tmp_result[f"A{j}"])) + str_a_bin
            str_b_bin = str(int(tmp_result[f"B{j}"])) + str_b_bin
            str_c_bin = str(int(tmp_result[f"C{j}"])) + str_c_bin
        str_c_bin = str(int(tmp_result[f"s{n_digits}"])) + str_c_bin
        result_a = int(str_a_bin, 2)
        result_b = int(str_b_bin, 2)
        result_c = int(str_c_bin, 2)
        # print(str_a_bin, str_b_bin, str_c_bin)
        if is_print_result:
            print(str(result_a), "+", str(result_b), "=", str(result_c))
    print("正しい結果　　：",
          sum(df_result[df_result.energy < 1e-5]["num_occurrences"]),
          "/", sum(df_result["num_occurrences"]))
    print("正しいパターン：", len(df_result[df_result.energy < 1e-5]))


if __name__ == "__main__":
    n_digits = 5
    bqm = generate_random_adder_qubo(n_digits)
    df_result = solve_dwave_mineng(bqm, num_reads=100)
    result_summary(df_result, n_digits)
