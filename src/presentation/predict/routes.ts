import { Router } from "express";
import { PredictController } from "./controller";
import { PredictService } from "../services";

export class PredictRoutes {
    static get routes(): Router {
        const router = Router();
        const controller = new PredictController( new PredictService() );

        router.post('/', controller.realizarPredict);

        return router;
    }
}