import argparse
import math
import os
import sqlite3
import time


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
	row = conn.execute(
		"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
		(table,),
	).fetchone()
	return row is not None


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
	return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def ensure_column(
	conn: sqlite3.Connection,
	table: str,
	column: str,
	sql_type: str,
) -> None:
	cols = get_table_columns(conn, table)
	if column not in cols:
		conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


def add_and_fill_columns(
	db_path: str,
	class_normalize: bool = False,
	tc_normalize: bool = False,
	inelas_normalize: bool = False,
	only_direction: bool = False,
	add_hlc: bool = False,
) -> None:
	if not os.path.exists(db_path):
		raise FileNotFoundError(f"DB not found: {db_path}")

	t0 = time.time()
	conn = sqlite3.connect(db_path)
	try:
		if not table_exists(conn, "truth"):
			raise RuntimeError("Table 'truth' not found in DB")
		if add_hlc and not table_exists(conn, "SplitInIcePulses"):
			raise RuntimeError("Table 'SplitInIcePulses' not found in DB")

		truth_cols = get_table_columns(conn, "truth")
		pulse_cols = get_table_columns(conn, "SplitInIcePulses") if add_hlc else []
		if not only_direction:
			if "this_db_osc_weight" not in truth_cols:
				raise RuntimeError(
					"Column 'this_db_osc_weight' not found in table 'truth'"
				)
			if "pid" not in truth_cols or "interaction_type" not in truth_cols:
				raise RuntimeError(
					"Columns 'pid' and/or 'interaction_type' not found in table 'truth'"
				)
		if add_hlc and "width" not in pulse_cols:
			raise RuntimeError(
				"Column 'width' not found in table 'SplitInIcePulses'"
			)

		if not only_direction:
			ensure_column(conn, "truth", "norm_this_db_osc_weight", "FLOAT")
			ensure_column(conn, "truth", "track_mu", "INTEGER")
		ensure_column(conn, "truth", "dir_x", "FLOAT")
		ensure_column(conn, "truth", "dir_y", "FLOAT")
		ensure_column(conn, "truth", "dir_z", "FLOAT")
		if class_normalize and not only_direction:
			ensure_column(conn, "truth", "norm_class_this_db_osc_weight", "FLOAT")
		if tc_normalize and not only_direction:
			ensure_column(conn, "truth", "norm_TC_this_db_osc_weight", "FLOAT")
		if inelas_normalize and not only_direction:
			ensure_column(conn, "truth", "norm_inelas_this_db_osc_weight", "FLOAT")
		if add_hlc:
			ensure_column(conn, "SplitInIcePulses", "hlc", "INTEGER")

		if "azimuth" not in truth_cols or "zenith" not in truth_cols:
			raise RuntimeError(
				"Columns 'azimuth' and/or 'zenith' not found in table 'truth'"
			)

		avg_weight = None
		if not only_direction:
			avg_weight = conn.execute(
				"SELECT AVG(this_db_osc_weight) FROM truth WHERE this_db_osc_weight IS NOT NULL"
			).fetchone()[0]

			if avg_weight is None:
				raise RuntimeError("Cannot normalize: all this_db_osc_weight values are NULL")
			if avg_weight == 0:
				raise RuntimeError("Cannot normalize: average this_db_osc_weight is 0")

			# Always fill the global normalization and track flag as before.
			conn.execute(
				"""
				UPDATE truth
				SET
					norm_this_db_osc_weight = CASE
						WHEN this_db_osc_weight IS NULL THEN NULL
						ELSE this_db_osc_weight / ?
					END,
					track_mu = CASE
						WHEN ABS(pid) = 14 AND interaction_type = 1 THEN 1
						ELSE 0
					END
				""",
				(avg_weight,),
			)

		# Fill direction vectors from azimuth/zenith.
		conn.execute("UPDATE truth SET dir_x = NULL, dir_y = NULL, dir_z = NULL")
		rows = conn.execute("SELECT rowid, azimuth, zenith FROM truth").fetchall()
		dir_updates = []
		for rowid, azimuth, zenith in rows:
			if azimuth is None or zenith is None:
				continue
			dir_x = float(math.sin(zenith) * math.cos(azimuth))
			dir_y = float(math.sin(zenith) * math.sin(azimuth))
			dir_z = float(math.cos(zenith))
			dir_updates.append((dir_x, dir_y, dir_z, rowid))

		for i in range(0, len(dir_updates), 10000):
			batch = dir_updates[i : i + 10000]
			conn.executemany(
				"UPDATE truth SET dir_x = ?, dir_y = ?, dir_z = ? WHERE rowid = ?",
				batch,
			)

		# If requested, compute normalization per-class into `norm_class_this_db_osc_weight`.
		# Classes (per user request):
		# - muons: pid = 13
		# - noise: pid = -1
		# - neutrinos: pid IN (-12, -14, -16, 12, 14, 16)
		if class_normalize and not only_direction:
			# compute averages for each class (ignore NULLs)
			avg_muon = conn.execute(
				"SELECT AVG(this_db_osc_weight) FROM truth WHERE pid = 13 AND this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			avg_noise = conn.execute(
				"SELECT AVG(this_db_osc_weight) FROM truth WHERE pid = -1 AND this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			avg_neutrino = conn.execute(
				"SELECT AVG(this_db_osc_weight) FROM truth WHERE pid IN (-12, -14, -16, 12, 14, 16) AND this_db_osc_weight IS NOT NULL"
			).fetchone()[0]

			# initialize column to NULL to make absent-class behavior explicit
			conn.execute("UPDATE truth SET norm_class_this_db_osc_weight = NULL")

			if avg_muon is not None and avg_muon != 0:
				conn.execute(
					"UPDATE truth SET norm_class_this_db_osc_weight = this_db_osc_weight / ? WHERE pid = 13 AND this_db_osc_weight IS NOT NULL",
					(avg_muon,),
				)
			if avg_noise is not None and avg_noise != 0:
				conn.execute(
					"UPDATE truth SET norm_class_this_db_osc_weight = this_db_osc_weight / ? WHERE pid = -1 AND this_db_osc_weight IS NOT NULL",
					(avg_noise,),
				)
			if avg_neutrino is not None and avg_neutrino != 0:
				conn.execute(
					"UPDATE truth SET norm_class_this_db_osc_weight = this_db_osc_weight / ? WHERE pid IN (-12, -14, -16, 12, 14, 16) AND this_db_osc_weight IS NOT NULL",
					(avg_neutrino,),
				)

		# If requested, normalize separately for track_mu=0 and track_mu=1 so
		# (1) overall normalized average stays 1 and
		# (2) total normalized weight is equal between both groups.
		if tc_normalize and not only_direction:
			sum_track0 = conn.execute(
				"SELECT SUM(norm_this_db_osc_weight) FROM truth WHERE track_mu = 0 AND norm_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			sum_track1 = conn.execute(
				"SELECT SUM(norm_this_db_osc_weight) FROM truth WHERE track_mu = 1 AND norm_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			n_tc_nonnull_total = conn.execute(
				"SELECT COUNT(*) FROM truth WHERE norm_this_db_osc_weight IS NOT NULL AND track_mu IN (0, 1)"
			).fetchone()[0]

			if sum_track0 is None or sum_track1 is None:
				raise RuntimeError(
					"Cannot TC-normalize: one of track_mu groups has no non-NULL norm_this_db_osc_weight"
				)
			if sum_track0 == 0 or sum_track1 == 0:
				raise RuntimeError(
					"Cannot TC-normalize: one of track_mu groups has sum(norm_this_db_osc_weight) = 0"
				)
			if n_tc_nonnull_total == 0:
				raise RuntimeError("Cannot TC-normalize: no non-NULL norm_this_db_osc_weight rows")

			# Let S0,S1 be group sums before TC scaling and N be total non-NULL rows.
			# We need x*S0 = y*S1 and (x*S0 + y*S1)/N = 1.
			# Therefore x*S0 = y*S1 = N/2.
			factor_track0 = float(n_tc_nonnull_total) / (2.0 * float(sum_track0))
			factor_track1 = float(n_tc_nonnull_total) / (2.0 * float(sum_track1))

			# Reset first so rows outside expected classes remain explicit NULL.
			conn.execute("UPDATE truth SET norm_TC_this_db_osc_weight = NULL")
			conn.execute(
				"UPDATE truth SET norm_TC_this_db_osc_weight = norm_this_db_osc_weight * ? WHERE track_mu = 0 AND norm_this_db_osc_weight IS NOT NULL",
				(factor_track0,),
			)
			conn.execute(
				"UPDATE truth SET norm_TC_this_db_osc_weight = norm_this_db_osc_weight * ? WHERE track_mu = 1 AND norm_this_db_osc_weight IS NOT NULL",
				(factor_track1,),
			)

		# If requested, normalize neutrino and anti-neutrino groups so
		# (1) overall normalized average stays 1 and
		# (2) total normalized weight is equal between sign groups.
		if inelas_normalize and not only_direction:
			sum_pid_pos = conn.execute(
				"SELECT SUM(norm_this_db_osc_weight) FROM truth WHERE pid IN (12, 14, 16) AND norm_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			sum_pid_neg = conn.execute(
				"SELECT SUM(norm_this_db_osc_weight) FROM truth WHERE pid IN (-12, -14, -16) AND norm_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			n_inelas_nonnull_total = conn.execute(
				"SELECT COUNT(*) FROM truth WHERE norm_this_db_osc_weight IS NOT NULL AND pid IN (12, 14, 16, -12, -14, -16)"
			).fetchone()[0]

			if sum_pid_pos is None or sum_pid_neg is None:
				raise RuntimeError(
					"Cannot inelas-normalize: one of PID-sign groups has no non-NULL norm_this_db_osc_weight"
				)
			if sum_pid_pos == 0 or sum_pid_neg == 0:
				raise RuntimeError(
					"Cannot inelas-normalize: one of PID-sign groups has sum(norm_this_db_osc_weight) = 0"
				)
			if n_inelas_nonnull_total == 0:
				raise RuntimeError("Cannot inelas-normalize: no non-NULL norm_this_db_osc_weight rows")

			# Let S+,S- be group sums before scaling and N be total non-NULL rows
			# included in this inelastic normalization.
			# We need x*S+ = y*S- and (x*S+ + y*S-)/N = 1.
			# Therefore x*S+ = y*S- = N/2.
			factor_pid_pos = float(n_inelas_nonnull_total) / (2.0 * float(sum_pid_pos))
			factor_pid_neg = float(n_inelas_nonnull_total) / (2.0 * float(sum_pid_neg))

			conn.execute("UPDATE truth SET norm_inelas_this_db_osc_weight = NULL")
			conn.execute(
				"UPDATE truth SET norm_inelas_this_db_osc_weight = norm_this_db_osc_weight * ? WHERE pid IN (12, 14, 16) AND norm_this_db_osc_weight IS NOT NULL",
				(factor_pid_pos,),
			)
			conn.execute(
				"UPDATE truth SET norm_inelas_this_db_osc_weight = norm_this_db_osc_weight * ? WHERE pid IN (-12, -14, -16) AND norm_this_db_osc_weight IS NOT NULL",
				(factor_pid_neg,),
			)

		# If requested, add/fill HLC flag for all pulses from pulse width.
		# width=1 -> hlc=1, width=8 -> hlc=0, anything else -> -1.
		if add_hlc:
			conn.execute(
				"""
				UPDATE SplitInIcePulses
				SET hlc = CASE
					WHEN width = 1 THEN 1
					WHEN width = 8 THEN 0
					ELSE -1
				END
				"""
			)
		conn.commit()

		n_rows = conn.execute("SELECT COUNT(*) FROM truth").fetchone()[0]
		n_norm_nonnull = None
		norm_avg = None
		n_track_mu = None
		if not only_direction:
			n_norm_nonnull = conn.execute(
				"SELECT COUNT(*) FROM truth WHERE norm_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			norm_avg = conn.execute(
				"SELECT AVG(norm_this_db_osc_weight) FROM truth WHERE norm_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			n_track_mu = conn.execute(
				"SELECT COUNT(*) FROM truth WHERE track_mu = 1"
			).fetchone()[0]
		n_dir_nonnull = conn.execute(
			"SELECT COUNT(*) FROM truth WHERE dir_x IS NOT NULL AND dir_y IS NOT NULL AND dir_z IS NOT NULL"
		).fetchone()[0]

		# class-normalization stats (if requested)
		n_class_norm_nonnull = None
		class_norm_details = None
		if class_normalize and not only_direction:
			n_class_norm_nonnull = conn.execute(
				"SELECT COUNT(*) FROM truth WHERE norm_class_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			class_norm_details = {
				"avg_muon": avg_muon,
				"avg_noise": avg_noise,
				"avg_neutrino": avg_neutrino,
			}

		# TC-normalization stats (if requested)
		n_tc_norm_nonnull = None
		tc_avg0 = None
		tc_avg1 = None
		tc_sum0 = None
		tc_sum1 = None
		tc_overall_avg = None
		tc_factor0 = None
		tc_factor1 = None
		inelas_norm_nonnull = None
		inelas_sum_pid_pos = None
		inelas_sum_pid_neg = None
		inelas_overall_avg = None
		inelas_avg_pid_pos = None
		inelas_avg_pid_neg = None
		inelas_factor_pid_pos = None
		inelas_factor_pid_neg = None
		n_pulses = None
		n_hlc_nonnull = None
		n_hlc_ones = None
		n_hlc_zeros = None
		if tc_normalize and not only_direction:
			n_tc_norm_nonnull = conn.execute(
				"SELECT COUNT(*) FROM truth WHERE norm_TC_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			tc_sum0 = conn.execute(
				"SELECT SUM(norm_TC_this_db_osc_weight) FROM truth WHERE track_mu = 0 AND norm_TC_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			tc_sum1 = conn.execute(
				"SELECT SUM(norm_TC_this_db_osc_weight) FROM truth WHERE track_mu = 1 AND norm_TC_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			tc_overall_avg = conn.execute(
				"SELECT AVG(norm_TC_this_db_osc_weight) FROM truth WHERE norm_TC_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			tc_avg0 = conn.execute(
				"SELECT AVG(norm_TC_this_db_osc_weight) FROM truth WHERE track_mu = 0 AND norm_TC_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			tc_avg1 = conn.execute(
				"SELECT AVG(norm_TC_this_db_osc_weight) FROM truth WHERE track_mu = 1 AND norm_TC_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			tc_factor0 = factor_track0
			tc_factor1 = factor_track1
		if inelas_normalize and not only_direction:
			inelas_norm_nonnull = conn.execute(
				"SELECT COUNT(*) FROM truth WHERE norm_inelas_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			inelas_sum_pid_pos = conn.execute(
				"SELECT SUM(norm_inelas_this_db_osc_weight) FROM truth WHERE pid IN (12, 14, 16) AND norm_inelas_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			inelas_sum_pid_neg = conn.execute(
				"SELECT SUM(norm_inelas_this_db_osc_weight) FROM truth WHERE pid IN (-12, -14, -16) AND norm_inelas_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			inelas_overall_avg = conn.execute(
				"SELECT AVG(norm_inelas_this_db_osc_weight) FROM truth WHERE norm_inelas_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			inelas_avg_pid_pos = conn.execute(
				"SELECT AVG(norm_inelas_this_db_osc_weight) FROM truth WHERE pid IN (12, 14, 16) AND norm_inelas_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			inelas_avg_pid_neg = conn.execute(
				"SELECT AVG(norm_inelas_this_db_osc_weight) FROM truth WHERE pid IN (-12, -14, -16) AND norm_inelas_this_db_osc_weight IS NOT NULL"
			).fetchone()[0]
			inelas_factor_pid_pos = factor_pid_pos
			inelas_factor_pid_neg = factor_pid_neg


		elapsed = time.time() - t0
		print(f"[DONE] Updated DB: {db_path}")
		print(f"  truth rows                    : {n_rows:,}")
		if not only_direction:
			print(f"  avg(this_db_osc_weight) used  : {avg_weight}")
			print(f"  non-null norm rows            : {n_norm_nonnull:,}")
			print(f"  avg(norm_this_db_osc_weight)  : {norm_avg}")
			print(f"  track_mu = 1 rows             : {n_track_mu:,}")
		print(f"  non-null direction rows       : {n_dir_nonnull:,}")
		if class_normalize and not only_direction:
			print(f"  non-null class-norm rows      : {n_class_norm_nonnull:,}")
			print(f"  class avg muon                : {class_norm_details['avg_muon']}")
			print(f"  class avg noise               : {class_norm_details['avg_noise']}")
			print(f"  class avg neutrino            : {class_norm_details['avg_neutrino']}")
		if tc_normalize and not only_direction:
			print(f"  non-null TC-norm rows         : {n_tc_norm_nonnull:,}")
			print(f"  TC factor(track_mu=0)         : {tc_factor0}")
			print(f"  TC factor(track_mu=1)         : {tc_factor1}")
			print(f"  TC sum(track_mu=0)            : {tc_sum0}")
			print(f"  TC sum(track_mu=1)            : {tc_sum1}")
			print(f"  TC avg(all non-null)          : {tc_overall_avg}")
			print(f"  TC avg(track_mu=0)            : {tc_avg0}")
			print(f"  TC avg(track_mu=1)            : {tc_avg1}")
		if inelas_normalize and not only_direction:
			print(f"  non-null inelas-norm rows     : {inelas_norm_nonnull:,}")
			print(f"  inelas factor(pid>0 set)      : {inelas_factor_pid_pos}")
			print(f"  inelas factor(pid<0 set)      : {inelas_factor_pid_neg}")
			print(f"  inelas sum(pid=12,14,16)      : {inelas_sum_pid_pos}")
			print(f"  inelas sum(pid=-12,-14,-16)   : {inelas_sum_pid_neg}")
			print(f"  inelas avg(all non-null)      : {inelas_overall_avg}")
			print(f"  inelas avg(pid=12,14,16)      : {inelas_avg_pid_pos}")
			print(f"  inelas avg(pid=-12,-14,-16)   : {inelas_avg_pid_neg}")
		print(f"  elapsed                       : {elapsed:.2f}s")
	finally:
		conn.close()


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Add/fill norm_this_db_osc_weight and track_mu in truth table of a SQLite DB."
		)
	)
	parser.add_argument(
		"db_path",
		help="Path to SQLite database containing truth.this_db_osc_weight",
	)
	parser.add_argument(
		"--class-normalize",
		dest="class_normalize",
		action="store_true",
		help=(
			"If set, also add and fill `norm_class_this_db_osc_weight`, normalizing "
			"this_db_osc_weight separately for muons (pid=13), noise (pid=-1), "
			"and neutrinos (pid in -12,-14,-16,12,14,16)."
		),
	)
	parser.add_argument(
		"--TC-normalize",
		dest="tc_normalize",
		action="store_true",
		help=(
			"If set, also add and fill `norm_TC_this_db_osc_weight` by scaling "
			"the globally normalized weight (`norm_this_db_osc_weight`) with one "
			"factor for track_mu=0 and one for track_mu=1 so that: "
			"(1) AVG(norm_TC_this_db_osc_weight)=1 over all non-NULL rows, and "
			"(2) SUM(norm_TC_this_db_osc_weight) is equal for track_mu=0 and "
			"track_mu=1."
		),
	)
	parser.add_argument(
		"--inelas-normalize",
		dest="inelas_normalize",
		action="store_true",
		help=(
			"If set, also add and fill `norm_inelas_this_db_osc_weight` by scaling "
			"the globally normalized weight (`norm_this_db_osc_weight`) with one "
			"factor for pid in (12,14,16) and one for pid in (-12,-14,-16) so "
			"that: (1) AVG(norm_inelas_this_db_osc_weight)=1 over included rows, "
			"and (2) SUM is equal between these two PID-sign groups."
		),
	)
	parser.add_argument(
		"--only-direction",
		dest="only_direction",
		action="store_true",
		help=(
			"Only add/fill `dir_x`, `dir_y`, `dir_z` from `azimuth` and `zenith`; "
			"do not modify any normalization or track columns."
		),
	)
	parser.add_argument(
		"--add-hlc",
		dest="add_hlc",
		action="store_true",
		help=(
			"If set, also add/fill `SplitInIcePulses.hlc` for all pulses using "
			"`width` (hlc=1 when width=1, hlc=0 when width=8, else -1)."
		),
	)
	args = parser.parse_args()

	if args.only_direction and (
		args.class_normalize or args.tc_normalize or args.inelas_normalize
	):
		raise RuntimeError(
			"--only-direction cannot be combined with --class-normalize, --TC-normalize, or --inelas-normalize"
		)

	add_and_fill_columns(
		args.db_path,
		class_normalize=args.class_normalize,
		tc_normalize=args.tc_normalize,
		inelas_normalize=args.inelas_normalize,
		only_direction=args.only_direction,
		add_hlc=args.add_hlc,
	)


if __name__ == "__main__":
	main()
