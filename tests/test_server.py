"""API tests. No network, no checkpoint — everything runs against the app in-process."""
import pytest

pytest.importorskip("fastapi", reason="fastapi is needed for the API tests")
pytest.importorskip("httpx", reason="httpx is needed by fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

from server.main import cors_origins, create_app  # noqa: E402

BOTS_ONLY = [{"type": "rule_based"}, {"type": "random"}]
HUMAN_VS_BOT = [{"type": "human"}, {"type": "rule_based"}]


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def create(client, players=HUMAN_VS_BOT, board="small_20", seed=17):
    response = client.post("/games", json={"board_config": board,
                                           "players": players, "seed": seed})
    assert response.status_code == 201, response.text
    return response.json()["game_id"]


class TestHealth:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"


class TestCreateGame:
    def test_returns_the_board_summary(self, client):
        body = client.post("/games", json={"board_config": "small_20",
                                           "players": HUMAN_VS_BOT}).json()
        assert body["num_territories"] == 20
        assert [p["type"] for p in body["players"]] == ["human", "rule_based"]
        assert body["players"][0]["is_human"]

    def test_seed_makes_the_deal_reproducible(self, client):
        first = client.get(f"/games/{create(client, seed=99)}").json()
        second = client.get(f"/games/{create(client, seed=99)}").json()
        assert ([t["owner"] for t in first["territories"]]
                == [t["owner"] for t in second["territories"]])

    @pytest.mark.parametrize("players", [
        [{"type": "human"}],                       # too few
        [{"type": "random"}] * 7,                  # too many
        [{"type": "wizard"}, {"type": "random"}],  # unknown agent
    ])
    def test_rejects_bad_player_lists(self, client, players):
        response = client.post("/games", json={"board_config": "small_20",
                                               "players": players})
        assert response.status_code == 422

    def test_rejects_more_players_than_the_board_seats(self, client):
        response = client.post("/games", json={
            "board_config": "small_20", "players": [{"type": "random"}] * 6})
        assert response.status_code == 400
        assert "4 players" in response.json()["detail"]

    def test_rejects_unknown_board(self, client):
        response = client.post("/games", json={"board_config": "atlantis",
                                               "players": HUMAN_VS_BOT})
        assert response.status_code == 422

    def test_all_bot_game_is_already_under_way(self, client):
        state = client.get(f"/games/{create(client, players=BOTS_ONLY)}").json()
        assert state["turn_number"] > 0


class TestGameState:
    def test_unknown_game_is_404(self, client):
        assert client.get("/games/deadbeef").status_code == 404
        assert client.get("/games/deadbeef/legal-actions").status_code == 404
        assert client.post("/games/deadbeef/step").status_code == 404

    def test_state_covers_the_whole_board(self, client):
        state = client.get(f"/games/{create(client)}").json()
        assert len(state["territories"]) == 20
        assert sum(p["territory_count"] for p in state["players"]) == 20
        assert all(t["owner"] in (0, 1) for t in state["territories"])

    def test_grid_metadata_only_on_grid_boards(self, client):
        assert client.get(f"/games/{create(client)}").json()["grid"] is None
        grid = client.get(f"/games/{create(client, board='grid_6x6')}").json()["grid"]
        assert grid == {"rows": 6, "cols": 6}


class TestLegalActions:
    def test_offers_only_playable_moves(self, client):
        game_id = create(client)
        body = client.get(f"/games/{game_id}/legal-actions").json()
        assert body["phase"] == "DRAFT"
        assert body["actions"]
        assert all(a["phase"] == "DRAFT" for a in body["actions"])

    def test_every_offered_action_is_accepted(self, client):
        game_id = create(client)
        action = client.get(f"/games/{game_id}/legal-actions").json()["actions"][0]
        response = client.post(f"/games/{game_id}/action", json={
            "phase": action["phase"], "src": action["src"],
            "dst": action["dst"], "troops": action["troops"]})
        assert response.status_code == 200, response.text

    def test_finished_game_offers_nothing(self, client):
        body = client.get(
            f"/games/{create(client, players=BOTS_ONLY)}/legal-actions").json()
        assert body["actions"] == []


class TestSubmitAction:
    def test_illegal_action_is_rejected(self, client):
        game_id = create(client)
        response = client.post(f"/games/{game_id}/action",
                               json={"phase": "ATTACK", "src": 0, "dst": 1, "troops": 99})
        assert response.status_code == 400
        assert "Illegal action" in response.json()["detail"]

    def test_unknown_phase_is_rejected(self, client):
        game_id = create(client)
        response = client.post(f"/games/{game_id}/action",
                               json={"phase": "SIESTA", "troops": -1})
        assert response.status_code == 400
        assert "Unknown phase" in response.json()["detail"]

    def test_cannot_move_for_a_bot(self, client):
        game_id = create(client, players=BOTS_ONLY)
        response = client.post(f"/games/{game_id}/action",
                               json={"phase": "DRAFT", "troops": -1})
        assert response.status_code == 400

    def test_bots_play_on_after_the_human_ends_a_turn(self, client):
        game_id = create(client)
        for phase in ("DRAFT", "ATTACK", "FORTIFY"):
            actions = client.get(f"/games/{game_id}/legal-actions").json()["actions"]
            end = next((a for a in actions if a["end_phase"]), actions[0])
            state = client.post(f"/games/{game_id}/action", json={
                "phase": end["phase"], "src": end["src"],
                "dst": end["dst"], "troops": end["troops"]}).json()
            if state["status"] == "finished":
                break
        # Control comes back to the human, never parked on the bot
        assert state["current_player"] == 0 or state["status"] == "finished"


class TestWebSocket:
    def test_pushes_state_on_connect(self, client):
        game_id = create(client)
        with client.websocket_connect(f"/ws/{game_id}") as ws:
            assert ws.receive_json()["game_id"] == game_id

    def test_any_message_gets_a_fresh_push(self, client):
        game_id = create(client)
        with client.websocket_connect(f"/ws/{game_id}") as ws:
            ws.receive_json()
            ws.send_text("ping")
            assert ws.receive_json()["phase"] in {"DRAFT", "ATTACK", "FORTIFY"}

    def test_unknown_game_is_closed(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/deadbeef") as ws:
                ws.receive_json()


class TestConfiguration:
    def test_cors_origins_default_to_the_dev_servers(self, monkeypatch):
        monkeypatch.delenv("RISK_CORS_ORIGINS", raising=False)
        assert "http://localhost:5173" in cors_origins()

    def test_cors_origins_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("RISK_CORS_ORIGINS", "https://a.test, https://b.test")
        assert cors_origins() == ["https://a.test", "https://b.test"]

    def test_sessions_are_evicted_rather_than_growing_without_bound(self):
        from server.game_manager import GameManager
        from server.schemas import GameCreateRequest, PlayerConfig
        manager = GameManager(max_sessions=3)
        request = GameCreateRequest(board_config="small_20",
                                    players=[PlayerConfig(type="human"),
                                             PlayerConfig(type="human")])
        ids = [manager.create_game(request)[0] for _ in range(6)]
        assert manager.get_session(ids[0]) is None
        assert manager.get_session(ids[-1]) is not None


class TestLegalActionsAreBounded:
    """
    The legal-action list is combinatorial in troop counts: a late-game fortify
    phase on classic_42 enumerates ~10^5 "move n troops from A to B" variants,
    several megabytes of JSON. The endpoint must cap what it ships.
    """

    def _late_game_session(self, client, troops=60):
        from engine.constants import Phase

        game_id = create(client, board="classic_42")
        manager = client.app.state.game_manager
        session = manager.get_session(game_id)
        state = session.state
        state.owners = [0] * (session.board.num_territories - 1) + [1]
        state.troops = [troops] * session.board.num_territories
        state.phase = Phase.FORTIFY
        state.current_player = 0
        return game_id, session

    def test_response_is_capped_and_says_so(self, client):
        game_id, _ = self._late_game_session(client)
        body = client.get(f"/games/{game_id}/legal-actions").json()
        assert body["truncated"] is True
        assert len(body["actions"]) == body["limit"] == 500

    def test_limit_is_honoured(self, client):
        game_id, _ = self._late_game_session(client)
        body = client.get(f"/games/{game_id}/legal-actions?limit=10").json()
        assert len(body["actions"]) == 10
        assert body["truncated"] is True

    def test_a_short_list_is_not_marked_truncated(self, client):
        game_id = create(client)
        body = client.get(f"/games/{game_id}/legal-actions").json()
        assert body["truncated"] is False
        assert len(body["actions"]) < body["limit"]

    def test_response_stays_small(self, client):
        """Before the cap this body was ~7 MB."""
        game_id, _ = self._late_game_session(client)
        response = client.get(f"/games/{game_id}/legal-actions")
        assert len(response.content) < 200_000

    @pytest.mark.parametrize("limit", [0, -1, 10 ** 9])
    def test_out_of_range_limits_are_refused(self, client, limit):
        game_id = create(client)
        assert client.get(
            f"/games/{game_id}/legal-actions?limit={limit}").status_code == 422

    def test_capped_actions_are_still_playable(self, client):
        game_id, session = self._late_game_session(client)
        body = client.get(f"/games/{game_id}/legal-actions?limit=25").json()
        for option in body["actions"]:
            from engine.constants import Phase
            from engine.rules import Action
            action = Action(phase=Phase[option["phase"]], src=option["src"],
                            dst=option["dst"], troops=option["troops"])
            assert session.engine.is_legal(session.state, action)


class TestConcurrentAccess:
    """
    FastAPI runs sync endpoints in a threadpool, so two requests for the same
    game genuinely overlap. Without a session lock both read the same state and
    the second write silently discards the first player's move.
    """

    def test_parallel_bot_steps_do_not_lose_moves(self, client):
        import concurrent.futures

        game_id = create(client, players=BOTS_ONLY)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = [f.result() for f in [
                pool.submit(client.post, f"/games/{game_id}/step") for _ in range(8)
            ]]
        assert all(r.status_code == 200 for r in results)
        state = client.get(f"/games/{game_id}").json()
        assert sum(p["territory_count"] for p in state["players"]) == 20

    def test_parallel_reads_return_consistent_states(self, client):
        import concurrent.futures

        game_id = create(client, players=BOTS_ONLY)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            bodies = [f.result().json() for f in
                      [pool.submit(client.get, f"/games/{game_id}") for _ in range(16)]]
        for body in bodies:
            # Territory counts must add up within a single response
            assert (sum(p["territory_count"] for p in body["players"])
                    == len(body["territories"]))


class TestServerBotBudget:
    def test_mcts_bots_are_budgeted_in_rollouts_not_seconds(self, client):
        """A wall-clock budget times a 5000-step ceiling is over an hour."""
        from server.game_manager import GameManager, SERVER_MCTS_SIMULATIONS
        from server.schemas import GameCreateRequest, PlayerConfig

        manager = GameManager()
        request = GameCreateRequest(board_config="small_20", seed=3,
                                    players=[PlayerConfig(type="human"),
                                             PlayerConfig(type="mcts")])
        _, session = manager.create_game(request)
        agent = session.agents[1]
        assert agent.num_simulations == SERVER_MCTS_SIMULATIONS

    def test_a_bad_checkpoint_is_a_client_error_not_a_500(self, client, tmp_path):
        broken = tmp_path / "broken.pt"
        broken.write_text("not a torch checkpoint")
        response = client.post("/games", json={
            "board_config": "small_20",
            "players": [{"type": "human"},
                        {"type": "rl", "checkpoint": str(broken)}]})
        assert response.status_code == 400

    def test_a_missing_checkpoint_is_a_client_error(self, client, tmp_path):
        response = client.post("/games", json={
            "board_config": "small_20",
            "players": [{"type": "human"},
                        {"type": "rl", "checkpoint": str(tmp_path / "nope.pt")}]})
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_a_directory_is_not_mistaken_for_a_checkpoint(self, client, tmp_path):
        response = client.post("/games", json={
            "board_config": "small_20",
            "players": [{"type": "human"},
                        {"type": "rl", "checkpoint": str(tmp_path)}]})
        assert response.status_code == 400
