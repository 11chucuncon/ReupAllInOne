from app.workers.tasks import app


if __name__ == "__main__":
    app.worker_main(argv=["worker", "--loglevel=info"])
